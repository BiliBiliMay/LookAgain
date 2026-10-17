"""Require complete paired coverage and valid cost records before aggregation."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from timelens.dataset.timelens_data import DATASET_DICT

CONDITIONS = ('baseline', 'lookagain_b15', 'fixed_search_b15')
DATASETS = ('charades-timelens', 'activitynet-timelens', 'qvhighlights-timelens')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = Path(args.root)
    out = root / ('smoke' if args.smoke else 'outputs')
    report = {}
    for dataset in DATASETS:
        if args.smoke:
            diagnostic = json.loads((out / f'{dataset}_diagnostic.json').read_text())
            assert len(diagnostic['verdicts']) == 2
            assert len(diagnostic['calls']) == 3
            assert diagnostic['crop'][0] > 0
        annos = DATASET_DICT[dataset].load_annos()
        annos.sort(key=lambda a: a['duration'], reverse=True)
        if args.smoke:
            annos = [annos[int((len(annos) - 1) * q)] for q in (0.5, 0.9)]
        keys = {f"{Path(a['video_path']).name}>>>{a['query']}>>>{a['span'][0]}" for a in annos}
        for condition in CONDITIONS:
            records = {}
            for index in range(1 if args.smoke else 120):
                marker = json.loads((out / 'completed' / f'{dataset}_{index}.json').read_text())
                assert marker['conditions'] == list(CONDITIONS)
                path = out / condition / f'{dataset}_{index}.jsonl'
                count = 0
                for line in path.read_text().splitlines():
                    item = json.loads(line)
                    assert len(item) == 1
                    key, value = next(iter(item.items()))
                    assert key not in records, (dataset, condition, 'duplicate', key)
                    records[key] = value
                    count += 1
                    calls = value['search_cost']['total_model_calls']
                    assert 1 <= calls <= (1 if condition == 'baseline' else 15)
                    assert all(not call['error'] for call in value['call_telemetry'])
                    assert value['model_revision'] == '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
                    assert value['condition'] == condition
                    assert all(math.isfinite(x) for span in value['timestamps'] for x in span)
                assert count == marker['queries']
            assert set(records) == keys, (dataset, condition, len(records), len(keys))
            valid = sum(any(0 <= s < e <= v['duration'] for s, e in v['timestamps']) for v in records.values())
            if args.smoke:
                assert valid > 0, (dataset, condition, 'no valid output')
                if condition != 'baseline':
                    assert any(v['search_cost']['total_model_calls'] > 1 for v in records.values())
            merged = out / condition / f'{dataset}.jsonl'
            merged.write_text(''.join(json.dumps({key: value}) + '\n' for key, value in records.items()))
            report[f'{dataset}/{condition}'] = {'queries': len(records), 'valid_predictions': valid}
            subprocess.run([sys.executable, 'evaluation/compute_metrics.py', '-f', str(merged)], check=True)
    (out / ('PASSED.json' if args.smoke else 'VERIFIED.json')).write_text(json.dumps(report, indent=2) + '\n')
    print('VALIDATION_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
