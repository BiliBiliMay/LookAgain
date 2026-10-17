"""Paired Qwen baseline / frozen LookAgain / fixed-search evaluation.

Outputs are checkpointed after every condition. Selection never uses annotations.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import torch
from evaluation.timelens_runner import TimeLensRunner
from evaluation.timelens_search import Interval, SearchConfig, run_timelens_search_sample
from evaluation.qwen_transfer_control import run_fixed_search
from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time

CONDITIONS = ('baseline', 'lookagain_b15', 'fixed_search_b15')


def key_for(anno):
    return f"{os.path.basename(anno['video_path'])}>>>{anno['query']}>>>{anno['span'][0]}"


class MeasuredRunner(TimeLensRunner):
    def __init__(self, args):
        super().__init__(args)
        self.phase = ''
        self.calls = []
        self.input_stats = {}

    def _build_inputs(self, anno, clip, fps_override=None, prompt_override=None):
        if self.phase == 'baseline':
            # Use the original one-call evaluator's full-video input path,
            # including its prompt without a search-specific crop prefix.
            inputs = GroundingDataset([anno], self.processor, self.args)[0]['inputs']
        else:
            inputs = super()._build_inputs(anno, clip, fps_override, prompt_override)
        grid = inputs.get('video_grid_thw')
        self.input_stats = {
            'input_tokens': int(inputs['input_ids'].numel()),
            'video_grid_thw': grid.tolist() if grid is not None else [],
        }
        return inputs

    def _generate_answer(self, anno, clip, fps_override=None, prompt_override=None,
                         max_new_tokens_override=None):
        started = time.monotonic()
        self.input_stats = {}
        error = None
        try:
            return super()._generate_answer(
                anno, clip, fps_override, prompt_override, max_new_tokens_override)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            self.calls.append({
                'phase': self.phase, 'crop': clip.as_list(),
                'fps': self.args.fps if self.phase == 'baseline' else fps_override,
                'wall_seconds': time.monotonic() - started,
                'error': error, **self.input_stats,
            })


def read_records(path):
    records = {}
    if not path.exists():
        return records
    # Recover only an incomplete final append after a killed job.
    with path.open('rb+') as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                handle.truncate(offset)
                print('RECOVER_INCOMPLETE_APPEND', path, offset, flush=True)
                break
            item = json.loads(line)
            assert len(item) == 1
            key, value = next(iter(item.items()))
            assert key not in records, (path, key)
            records[key] = value
    return records


def append_record(path, key, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps({key: value}) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def baseline_prediction(anno, runner, unit):
    answer = runner._generate_answer(anno, Interval(0, float(anno['duration'])),
                                     fps_override=runner.args.fps)
    timestamps = extract_time(answer)
    if not timestamps:
        timestamps = [[float(anno['duration']) + 10, float(anno['duration']) + 20]]
    timestamps = [[round(s / unit) * unit, round(e / unit) * unit] for s, e in timestamps]
    return dict(timestamps=timestamps, answers=answer, duration=float(anno['duration']),
                search_cost={'total_model_calls': 1}, selected_source='baseline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--dataset', required=True, choices=tuple(DATASET_DICT))
    parser.add_argument('--chunk', type=int, default=120)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    code_manifest = json.loads((root / 'code_manifest.json').read_text())
    for relative, expected in code_manifest.items():
        assert hashlib.sha256(Path(relative).read_bytes()).hexdigest() == expected, relative
    random.seed(42)
    torch.manual_seed(42)
    config = SearchConfig(search_mode='lookagain_astar', max_total_calls=15,
                          lookagain_controller='guided', lookagain_anchor_period=3,
                          critic_mode='self', critic_calls=2)
    runtime = argparse.Namespace(
        model_path=(root / 'model_path.txt').read_text().strip(),
        dtype='auto', attn_implementation='sdpa', device='balanced',
        max_gpu_memory_gib=14, cpu_offload_gib=120,
        offload_folder=str(root / 'offload' / os.environ.get('SLURM_JOB_ID', 'local')),
        offload_state_dict='true', tp_plan='none', use_cache='auto',
        sdpa_backend='efficient', fps=2, zoom_fps=4, min_tokens=64,
        total_tokens=14336, max_new_tokens=512, skip_video_errors='false',
    )
    dataset = DATASET_DICT[args.dataset]
    annos = dataset.load_annos()
    assert len({key_for(a) for a in annos}) == len(annos)
    annos.sort(key=lambda a: a['duration'], reverse=True)
    if args.smoke:
        # Duration-selected execution checks, independent of model quality.
        annos = [annos[int((len(annos) - 1) * q)] for q in (0.5, 0.9)]
    else:
        assert 0 <= args.index < args.chunk
        annos = annos[args.index::args.chunk]
    output_root = root / ('smoke' if args.smoke else 'outputs')
    paths = {c: output_root / c / f'{args.dataset}_{args.index}.jsonl' for c in CONDITIONS}
    records = {c: read_records(path) for c, path in paths.items()}
    expected_keys = {key_for(a) for a in annos}
    for condition in CONDITIONS:
        assert set(records[condition]).issubset(expected_keys)
    print('CONFIG', json.dumps({'runtime': vars(runtime), 'search': asdict(config),
                               'dataset': args.dataset, 'index': args.index,
                               'queries': len(annos), 'smoke': args.smoke}), flush=True)
    runner = MeasuredRunner(runtime)
    for number, anno in enumerate(annos, 1):
        key = key_for(anno)
        inference_anno = {k: v for k, v in anno.items() if k != 'span'}
        for condition in CONDITIONS:
            if key in records[condition]:
                continue
            runner.phase = condition
            runner.calls = []
            started = time.monotonic()
            if condition == 'baseline':
                value = baseline_prediction(inference_anno, runner, 1.0)
            else:
                baseline = records['baseline'][key]
                fn = run_timelens_search_sample if condition == 'lookagain_b15' else run_fixed_search
                result = fn(inference_anno, baseline, runner, config, unit=1.0)
                value = result.to_prediction_value(float(anno['duration']), config)
                # Count the actual baseline generation even when parsing failed.
                value['search_cost']['baseline_calls'] = 1
                value['search_cost']['total_model_calls'] = 1 + len(runner.calls)
                assert value['search_cost']['model_calls'] == len(runner.calls)
                assert value['search_cost']['total_model_calls'] <= 15
            telemetry = list(runner.calls)
            baseline_telemetry = [] if condition == 'baseline' else records['baseline'][key]['call_telemetry']
            value.update(
                condition=condition, model_id='Qwen/Qwen3-VL-8B-Instruct',
                model_revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
                call_telemetry=telemetry,
                condition_wall_seconds=time.monotonic() - started,
                total_input_tokens_including_baseline=sum(c.get('input_tokens', 0) for c in telemetry + baseline_telemetry),
                total_call_wall_seconds_including_baseline=sum(c['wall_seconds'] for c in telemetry + baseline_telemetry),
                peak_cuda_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())],
            )
            append_record(paths[condition], key, value)
            records[condition][key] = value
            print('PROGRESS', number, len(annos), condition, key,
                  value['timestamps'], value['search_cost']['total_model_calls'], flush=True)
    for condition in CONDITIONS:
        assert set(records[condition]) == expected_keys
    if args.smoke:
        diagnostic_path = output_root / f'{args.dataset}_diagnostic.json'
        if not diagnostic_path.exists():
            probe_anno = {k: v for k, v in annos[0].items() if k != 'span'}
            duration = float(probe_anno['duration'])
            runner.phase = 'smoke_diagnostic'
            runner.calls = []
            clip = Interval(duration * 0.5, duration * 0.75)
            prediction = runner.predict_interval(probe_anno, clip, fps_override=4)
            pair = (Interval(duration * 0.25, duration * 0.5), clip)
            verdicts = [runner.verify_pair(probe_anno, *pair, fps_override=4),
                        runner.verify_pair(probe_anno, *reversed(pair), fps_override=4)]
            # Require an explicit structured verdict, not parser defaults.
            for verdict in verdicts:
                payloads = []
                for match in re.findall(r'\{.*?\}', verdict.answer, flags=re.DOTALL):
                    try:
                        payloads.append(json.loads(match))
                    except json.JSONDecodeError:
                        pass
                assert any(p.get('choice') in ('A', 'B', 'UNCERTAIN') and
                           isinstance(p.get('confidence'), (int, float))
                           for p in payloads), verdict.answer
            assert all(c.get('input_tokens', 0) > 0 and c.get('video_grid_thw')
                       for c in runner.calls)
            diagnostic_path.write_text(json.dumps({
                'crop': clip.as_list(), 'prediction': asdict(prediction),
                'verdicts': [asdict(v) for v in verdicts], 'calls': runner.calls,
            }, indent=2) + '\n')
    done = output_root / 'completed' / f'{args.dataset}_{args.index}.json'
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps({'queries': len(annos), 'conditions': CONDITIONS,
                                'dataset': args.dataset, 'index': args.index}) + '\n')
    print('CHUNK_COMPLETE', done, flush=True)


if __name__ == '__main__':
    main()
