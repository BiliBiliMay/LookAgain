# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.
# Original code copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import argparse
import gc
import os
from contextlib import nullcontext

import nncore
import torch
import torch.distributed as dist
from nncore.engine import set_random_seed
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
SDPA_BACKEND_MAP = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}
FORCE_REPEAT_KV_FOR_SDPA = os.environ.get("FORCE_REPEAT_KV_FOR_SDPA", "0") == "1"


def format_gib(value):
    if value is None:
        return None
    if float(value).is_integer():
        return f"{int(value)}GiB"
    return f"{value}GiB"


def resolve_device_settings(args):
    capability = None
    device_name = "cpu"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)

    if args.dtype == "auto":
        if capability is None:
            dtype = torch.float32
        elif capability[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = DTYPE_MAP[args.dtype]

    if args.attn_implementation == "auto":
        if capability is not None and capability[0] >= 8:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
    else:
        attn_implementation = args.attn_implementation

    print(
        "Resolved runtime config: "
        f"device={device_name}, capability={capability}, "
        f"dtype={dtype}, attn_implementation={attn_implementation}"
    )
    return dtype, attn_implementation


def repeat_kv_to_match_heads(hidden_states, target_heads):
    current_heads = hidden_states.shape[1]
    if current_heads == target_heads:
        return hidden_states
    if target_heads % current_heads != 0:
        raise ValueError(
            f"Cannot expand KV heads from {current_heads} to {target_heads}."
        )
    n_rep = target_heads // current_heads
    batch, _, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, current_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, target_heads, slen, head_dim)


def patched_sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        print(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`; "
            "falling back may be required."
        )

    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if FORCE_REPEAT_KV_FOR_SDPA or not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv_to_match_heads(key, query.shape[1])
            value = repeat_kv_to_match_heads(value, query.shape[1])
        else:
            sdpa_kwargs = {"enable_gqa": True}
    elif query.shape[1] != key.shape[1]:
        key = repeat_kv_to_match_heads(key, query.shape[1])
        value = repeat_kv_to_match_heads(value, query.shape[1])

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    if is_causal is None:
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(
            module, "is_causal", True
        )

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def get_distributed_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def resolve_model_parallel_args(args):
    if args.tp_plan != "none":
        return {
            "tp_plan": args.tp_plan,
        }

    model_parallel_args = {
        "device_map": args.device,
    }
    max_memory = {}
    if args.max_gpu_memory_gib is not None and torch.cuda.is_available():
        gpu_budget = format_gib(args.max_gpu_memory_gib)
        for gpu_idx in range(torch.cuda.device_count()):
            max_memory[gpu_idx] = gpu_budget
    if args.cpu_offload_gib is not None:
        max_memory["cpu"] = format_gib(args.cpu_offload_gib)
    if max_memory:
        model_parallel_args["max_memory"] = max_memory
    if args.offload_folder:
        os.makedirs(args.offload_folder, exist_ok=True)
        model_parallel_args["offload_folder"] = args.offload_folder
    if args.offload_state_dict != "auto":
        model_parallel_args["offload_state_dict"] = args.offload_state_dict == "true"
    return model_parallel_args


def normalize_model_device(device):
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    if isinstance(device, str):
        if device in ("cpu", "disk"):
            return None
        if device.isdigit():
            return torch.device(f"cuda:{device}")
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    return None


def resolve_input_device(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        preferred_keys = (
            "language_model.model.embed_tokens",
            "model.embed_tokens",
            "embed_tokens",
            "language_model",
            "model",
            "",
        )
        for key in preferred_keys:
            if key in hf_device_map:
                device = normalize_model_device(hf_device_map[key])
                if device is not None:
                    return device
        for device_value in hf_device_map.values():
            device = normalize_model_device(device_value)
            if device is not None:
                return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def summarize_device_map(model):
    hf_device_map = getattr(model, "hf_device_map", None)
    if not hf_device_map:
        return
    summary = {}
    for device_value in hf_device_map.values():
        key = str(device_value)
        summary[key] = summary.get(key, 0) + 1
    print(f"HF device_map summary: {summary}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True, help="Output prediction path")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)
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
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit the number of examples after chunking. Useful for smoke tests.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=10,
        help="Number of DataLoader workers used to preprocess video samples.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of prefetched batches per worker when num_workers > 0.",
    )
    parser.add_argument(
        "--pin_memory",
        default="auto",
        choices=("auto", "true", "false"),
        help="Whether to pin host memory in the DataLoader.",
    )
    parser.add_argument(
        "--skip_video_errors",
        default="false",
        choices=("true", "false"),
        help="Skip unreadable video samples and emit zero-IoU placeholder predictions instead of aborting.",
    )

    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max_gpu_memory_gib",
        type=float,
        default=None,
        help="Per-GPU memory cap in GiB when using device_map evaluation.",
    )
    parser.add_argument(
        "--cpu_offload_gib",
        type=float,
        default=None,
        help="CPU memory budget in GiB for accelerate offload when using device_map evaluation.",
    )
    parser.add_argument(
        "--offload_folder",
        default=None,
        help="Directory used for accelerate offload artifacts when using device_map evaluation.",
    )
    parser.add_argument(
        "--offload_state_dict",
        default="auto",
        choices=("auto", "true", "false"),
        help="Override accelerate state-dict offload during model load.",
    )
    parser.add_argument(
        "--tp_plan",
        default="none",
        choices=("none", "auto"),
        help="Enable torch tensor parallelism. Requires launching with torchrun.",
    )
    parser.add_argument(
        "--use_cache",
        default="auto",
        choices=("auto", "true", "false"),
        help="Override generation KV cache usage. Useful when memory is tight.",
    )
    parser.add_argument(
        "--sdpa_backend",
        default="auto",
        choices=("auto", "flash", "efficient", "math"),
        help="Optionally force a specific PyTorch SDPA backend during generation.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=1,
        help="Number of chunks to split the dataset for distributed evaluation. Default is 1.",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    ALL_ATTENTION_FUNCTIONS["sdpa"] = patched_sdpa_attention_forward

    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"

    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )

    dtype, attn_implementation = resolve_device_settings(args)
    model_parallel_args = resolve_model_parallel_args(args)
    if args.tp_plan == "none":
        print(
            "Device-map runtime config: "
            f"device_map={args.device}, "
            f"max_gpu_memory_gib={args.max_gpu_memory_gib}, "
            f"cpu_offload_gib={args.cpu_offload_gib}, "
            f"offload_folder={args.offload_folder}, "
            f"offload_state_dict={args.offload_state_dict}"
        )

    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
        **model_parallel_args,
    ).eval()

    rank, world_size, local_rank = get_distributed_info()
    print(
        f"Distributed runtime: rank={rank}, world_size={world_size}, local_rank={local_rank}, "
        f"tp_plan={args.tp_plan}"
    )
    summarize_device_map(model)
    input_device = resolve_input_device(model)
    print(f"Resolved model input device: {input_device}")

    # Load processor (model-specific)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,  # For Video Processing, we do not need to resize the video frames again in the processor
        use_fast=False,
        trust_remote_code=True,
    )
    # Load dataset
    dataset_class = DATASET_DICT[args.dataset]
    annos = dataset_class.load_annos(split=args.split)

    # Sort by video length in descending order
    # 1. balance the video length for each GPU
    # 2. long videos are more likely to cause OOM, so we put them first
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    if args.max_examples is not None:
        annos = annos[: args.max_examples]
        print(f"Limiting evaluation to {len(annos)} examples for smoke testing.")

    dataset = GroundingDataset(annos, processor, args)
    pin_memory = torch.cuda.is_available() if args.pin_memory == "auto" else args.pin_memory == "true"
    data_loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: x[0],
    )
    if args.num_workers > 0:
        data_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    print(
        "DataLoader config: "
        f"num_workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor if args.num_workers > 0 else 'n/a'}, "
        f"pin_memory={pin_memory}"
    )
    data_loader = DataLoader(**data_loader_kwargs)

    dumps = []
    for data in nncore.ProgressBar(data_loader):
        anno = data["anno"]
        video_path = anno["video_path"]
        query = anno["query"]
        duration = anno["duration"]
        span = anno["span"]  # ground truth time span

        if data["inputs"] is None:
            error = data.get("error", "unknown video processing error")
            print(f"Skipping unreadable video sample: video_path={video_path}, query={query}, error={error}")
            timestamps = [[duration + 10, duration + 20]]
            answers = f"SKIPPED_DUE_TO_VIDEO_ERROR: {error}"
        else:
            non_blocking = input_device.type == "cuda"
            inputs = data["inputs"].to(input_device, non_blocking=non_blocking)

            generate_kwargs = dict(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=512,
            )
            if args.use_cache != "auto":
                generate_kwargs["use_cache"] = args.use_cache == "true"

            if args.attn_implementation == "sdpa" and args.sdpa_backend != "auto":
                generate_context = sdpa_kernel([SDPA_BACKEND_MAP[args.sdpa_backend]])
            else:
                generate_context = nullcontext()

            with generate_context:
                output_ids = model.generate(**generate_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            answers = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            answers = answers[0]

            # Parse the answer
            timestamps = extract_time(answers)
            if len(timestamps) != 0:
                print(f"Extracted timestamps: {timestamps}")
            else:
                print("No timestamps extracted, answer might be invalid. Answer:", answers)
                timestamps = [[duration + 10, duration + 20]]

        # Round timestamps to units
        unit = getattr(dataset_class, "UNIT", 1.0)
        timestamps = [
            [
                round(start / unit) * unit,
                round(end / unit) * unit,
            ]
            for start, end in timestamps
        ]

        # Save the inference results
        video_name = os.path.basename(video_path)
        if type(span[0]) is list or type(span[0]) is tuple:
            span = span[0]

        dump = {
            f"{video_name}>>>{query}>>>{span}": {
                "timestamps": timestamps,  # the extracted time span prediction from the model
                "answers": answers,  # the full answer from the model
                "duration": duration,  # save the video duration
            }
        }

        print(
            f"video_path: {video_path}, query: {query}, duration: {duration}, "
            f"answer: {answers}, extracted timestamps: {timestamps}"
        )

        dumps.append(dump)

        if data["inputs"] is not None:
            del output_ids
            del generated_ids_trimmed
            del generate_kwargs
            del inputs
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        nncore.dump(dumps, pred_path)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
