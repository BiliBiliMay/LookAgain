#!/usr/bin/env python
# ActiveZoom-VTG evaluation entrypoint for TimeLens.

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import torch.distributed as dist

from evaluation.activezoom import ZoomConfig, run_activezoom_sample
from evaluation.eval_dataloader import get_distributed_info
from evaluation.timelens_runner import TimeLensRunner
from timelens.dataset.timelens_data import DATASET_DICT


def read_jsonl_return_dict(file_path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with open(file_path, "r") as reader:
        for line in reader:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Each line in the JSONL file should be a dictionary.")
            data.update(item)
    return data


def read_predictions(path: str) -> Dict[str, Any]:
    if path.endswith(".jsonl"):
        return read_jsonl_return_dict(path)
    if path.endswith(".json"):
        with open(path, "r") as reader:
            data = json.load(reader)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a prediction dictionary in {path}")
        return data
    raise ValueError("baseline_jsonl must end with .jsonl or .json")


def write_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as writer:
        for item in items:
            writer.write(json.dumps(item) + "\n")


def sample_key(anno: Dict[str, Any]) -> str:
    video_name = os.path.basename(anno["video_path"])
    span = anno["span"]
    if isinstance(span[0], (list, tuple)):
        span = span[0]
    return f"{video_name}>>>{anno['query']}>>>{span}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--baseline_jsonl", required=True, help="One-shot TimeLens prediction JSONL")
    parser.add_argument("--pred_path", required=True, help="Output prediction prefix")
    parser.add_argument("--model_path", required=True, help="Path to the TimeLens model")
    parser.add_argument("--split", default="test")

    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--zoom_fps", type=int, default=4)
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "bfloat16", "float16", "float32"),
    )
    parser.add_argument(
        "--attn_implementation",
        default="auto",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_gpu_memory_gib", type=float, default=None)
    parser.add_argument("--cpu_offload_gib", type=float, default=None)
    parser.add_argument("--offload_folder", default=None)
    parser.add_argument(
        "--offload_state_dict",
        default="auto",
        choices=("auto", "true", "false"),
    )
    parser.add_argument(
        "--tp_plan",
        default="none",
        choices=("none", "auto"),
    )
    parser.add_argument(
        "--use_cache",
        default="auto",
        choices=("auto", "true", "false"),
    )
    parser.add_argument(
        "--sdpa_backend",
        default="auto",
        choices=("auto", "flash", "efficient", "math"),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--pin_memory",
        default="false",
        choices=("auto", "true", "false"),
    )
    parser.add_argument(
        "--skip_video_errors",
        default="false",
        choices=("true", "false"),
    )
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--activezoom_iterations", type=int, default=2)
    parser.add_argument("--activezoom_top_k", type=int, default=3)
    parser.add_argument("--activezoom_min_window_seconds", type=float, default=8.0)
    parser.add_argument("--activezoom_map_threshold", type=float, default=0.35)
    parser.add_argument("--activezoom_broad_replace_threshold", type=float, default=0.55)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    pred_path = f"{args.pred_path}_{args.index}.jsonl"
    dataset_class = DATASET_DICT[args.dataset]
    unit = getattr(dataset_class, "UNIT", 1.0)
    baseline_predictions = read_predictions(args.baseline_jsonl)
    config = ZoomConfig(
        iterations=args.activezoom_iterations,
        top_k=args.activezoom_top_k,
        min_window_seconds=args.activezoom_min_window_seconds,
        map_threshold=args.activezoom_map_threshold,
        broad_replace_threshold=args.activezoom_broad_replace_threshold,
    )

    annos = dataset_class.load_annos(split=args.split)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    if args.max_examples is not None:
        annos = annos[: args.max_examples]

    rank, world_size, local_rank = get_distributed_info()
    print(
        f"ActiveZoom dataset={args.dataset} split={args.split} "
        f"chunk={args.chunk} index={args.index} output={pred_path}"
    )
    print(
        f"Distributed runtime: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, tp_plan={args.tp_plan}"
    )
    print(
        f"ActiveZoom config: iterations={config.iterations}, top_k={config.top_k}, "
        f"zoom_fps={args.zoom_fps}, min_window={config.min_window_seconds}, unit={unit}"
    )

    runner = TimeLensRunner(args)
    dumps: List[Dict[str, Any]] = []
    missing_baseline = 0

    for sample_index, anno in enumerate(annos, start=1):
        key = sample_key(anno)
        baseline_pred = baseline_predictions.get(key)
        if baseline_pred is None:
            missing_baseline += 1
            print(f"Missing baseline prediction for key: {key}")

        result = run_activezoom_sample(
            anno=anno,
            baseline_pred=baseline_pred,
            runner=runner,
            config=config,
            unit=unit,
        )
        timestamps = [result.interval.as_list()]
        dump = {
            key: {
                "timestamps": timestamps,
                "answers": result.answer,
                "duration": anno["duration"],
                "activezoom": result.to_metadata(config),
            }
        }
        if rank == 0:
            dumps.append(dump)

        print(
            f"[{sample_index}/{len(annos)}] key={key} "
            f"selected={result.selected_source} fallback={result.fallback_used} "
            f"confidence={result.selected_confidence:.4f} timestamps={timestamps}"
        )

    if missing_baseline:
        print(f"Missing baseline predictions: {missing_baseline}/{len(annos)}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        write_jsonl(dumps, pred_path)
        print(f"Wrote {len(dumps)} ActiveZoom predictions to {pred_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
