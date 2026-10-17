#!/usr/bin/env python
"""TimeLens temporal search evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import torch.distributed as dist

from evaluation.timelens_runner import TimeLensRunner
from evaluation.timelens_search import SearchConfig, run_timelens_search_sample
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


def get_distributed_info() -> Tuple[int, int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--baseline_jsonl", required=True, help="One-shot TimeLens prediction JSONL")
    parser.add_argument("--pred_path", required=True, help="Output prediction prefix")
    parser.add_argument("--model_path", required=True, help="Path to the TimeLens model")
    parser.add_argument("--split", default="test")

    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--zoom_fps", type=int, default=4)
    parser.add_argument("--dense_zoom_fps", type=int, default=8)
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
    parser.add_argument("--tp_plan", default="none", choices=("none", "auto"))
    parser.add_argument("--use_cache", default="auto", choices=("auto", "true", "false"))
    parser.add_argument(
        "--sdpa_backend",
        default="auto",
        choices=("auto", "flash", "efficient", "math"),
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--pin_memory", default="false", choices=("auto", "true", "false"))
    parser.add_argument("--skip_video_errors", default="false", choices=("true", "false"))
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--search_mode",
        default="budgeted_graph",
        choices=("lookagain_astar", "budgeted_graph", "exhaustive_refine", "coarse_only"),
    )
    parser.add_argument("--coarse_grid_size", type=int, default=9)
    parser.add_argument("--coarse_window_fraction", type=float, default=0.25)
    parser.add_argument("--refine_rounds", type=int, default=2)
    parser.add_argument("--top_k_stage1", type=int, default=5)
    parser.add_argument("--top_k_refine", type=int, default=5)
    parser.add_argument("--min_window_seconds", type=float, default=4.0)
    parser.add_argument("--tiny_event_window_seconds", type=float, default=8.0)
    parser.add_argument("--diffuse_confidence_threshold", type=float, default=0.30)
    parser.add_argument("--broad_baseline_fraction", type=float, default=0.60)
    parser.add_argument("--broad_replace_threshold", type=float, default=0.55)
    parser.add_argument("--max_trace_answer_chars", type=int, default=256)
    parser.add_argument("--max_total_calls", type=int, default=15)
    parser.add_argument("--graph_seed_count", type=int, default=5)
    parser.add_argument("--graph_beam_width", type=int, default=3)
    parser.add_argument("--graph_merge_iou", type=float, default=0.92)
    parser.add_argument("--graph_support_iou", type=float, default=0.55)
    parser.add_argument(
        "--lookagain_controller",
        default="guided",
        choices=("ucs", "guided"),
    )
    parser.add_argument("--lookagain_anchor_period", type=int, default=3)
    parser.add_argument("--certificate_boundary_iou", type=float, default=0.55)
    parser.add_argument("--certificate_support_iou", type=float, default=0.55)
    parser.add_argument(
        "--certificate_contradiction_weight", type=float, default=0.45
    )
    parser.add_argument("--max_controller_trace_steps", type=int, default=256)
    parser.add_argument("--critic_mode", default="self", choices=("none", "self"))
    parser.add_argument("--critic_calls", type=int, default=2)
    parser.add_argument("--critic_fps", type=int, default=4)
    parser.add_argument("--critic_min_confidence", type=float, default=0.55)

    return parser.parse_args()


def build_search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        search_mode=args.search_mode,
        coarse_grid_size=args.coarse_grid_size,
        coarse_window_fraction=args.coarse_window_fraction,
        refine_rounds=args.refine_rounds,
        top_k_stage1=args.top_k_stage1,
        top_k_refine=args.top_k_refine,
        min_window_seconds=args.min_window_seconds,
        tiny_event_window_seconds=args.tiny_event_window_seconds,
        fps=args.fps,
        zoom_fps=args.zoom_fps,
        dense_zoom_fps=args.dense_zoom_fps,
        diffuse_confidence_threshold=args.diffuse_confidence_threshold,
        broad_baseline_fraction=args.broad_baseline_fraction,
        broad_replace_threshold=args.broad_replace_threshold,
        max_trace_answer_chars=args.max_trace_answer_chars,
        max_total_calls=args.max_total_calls,
        graph_seed_count=args.graph_seed_count,
        graph_beam_width=args.graph_beam_width,
        graph_merge_iou=args.graph_merge_iou,
        graph_support_iou=args.graph_support_iou,
        lookagain_controller=args.lookagain_controller,
        lookagain_anchor_period=args.lookagain_anchor_period,
        certificate_boundary_iou=args.certificate_boundary_iou,
        certificate_support_iou=args.certificate_support_iou,
        certificate_contradiction_weight=args.certificate_contradiction_weight,
        max_controller_trace_steps=args.max_controller_trace_steps,
        critic_mode=args.critic_mode,
        critic_calls=args.critic_calls,
        critic_fps=args.critic_fps,
        critic_min_confidence=args.critic_min_confidence,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    pred_path = f"{args.pred_path}_{args.index}.jsonl"
    dataset_class = DATASET_DICT[args.dataset]
    unit = getattr(dataset_class, "UNIT", 1.0)
    baseline_predictions = read_predictions(args.baseline_jsonl)
    config = build_search_config(args)

    annos = dataset_class.load_annos(split=args.split)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    if args.max_examples is not None:
        annos.sort(key=lambda x: x["duration"])
        annos = annos[: args.max_examples]

    rank, world_size, local_rank = get_distributed_info()
    print(
        f"TimeLensSearch dataset={args.dataset} split={args.split} "
        f"chunk={args.chunk} index={args.index} output={pred_path}"
    )
    print(
        f"Distributed runtime: rank={rank}, world_size={world_size}, "
        f"local_rank={local_rank}, tp_plan={args.tp_plan}"
    )
    print(
        f"Search config: mode={config.search_mode}, coarse_grid={config.coarse_grid_size}, "
        f"coarse_fraction={config.coarse_window_fraction}, refine_rounds={config.refine_rounds}, "
        f"top_k_stage1={config.top_k_stage1}, top_k_refine={config.top_k_refine}, "
        f"fps={config.fps}, zoom_fps={config.zoom_fps}, dense_zoom_fps={config.dense_zoom_fps}, "
        f"max_total_calls={config.max_total_calls}, critic={config.critic_mode}/{config.critic_calls}, "
        f"lookagain_controller={config.lookagain_controller}, "
        f"anchor_period={config.lookagain_anchor_period}, "
        f"unit={unit}"
    )
    if annos:
        durations = [float(anno["duration"]) for anno in annos]
        print(
            f"Selected {len(annos)} examples; "
            f"duration_range={min(durations):.3f}-{max(durations):.3f}; "
            f"max_new_tokens={args.max_new_tokens}"
        )
    else:
        print("Selected 0 examples.")

    runner = TimeLensRunner(args)
    dumps: List[Dict[str, Any]] = []
    missing_baseline = 0

    for sample_index, anno in enumerate(annos, start=1):
        key = sample_key(anno)
        baseline_pred = baseline_predictions.get(key)
        if baseline_pred is None:
            missing_baseline += 1
            print(f"Missing baseline prediction for key: {key}")

        result = run_timelens_search_sample(
            anno=anno,
            baseline_pred=baseline_pred,
            runner=runner,
            config=config,
            unit=unit,
        )
        dump = {
            key: result.to_prediction_value(
                duration=float(anno["duration"]),
                config=config,
            )
        }
        if rank == 0:
            dumps.append(dump)

        print(
            f"[{sample_index}/{len(annos)}] key={key} "
            f"selected={result.selected_source} fallback={result.fallback_used} "
            f"confidence={result.selected_confidence:.4f} "
            f"total_calls={result.search_cost.get('total_model_calls', result.search_cost.get('model_calls', 0))} "
            f"timestamps={[result.interval.as_list()]}"
        )

    if missing_baseline:
        print(f"Missing baseline predictions: {missing_baseline}/{len(annos)}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        write_jsonl(dumps, pred_path)
        print(f"Wrote {len(dumps)} TimeLens search predictions to {pred_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
