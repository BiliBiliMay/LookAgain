"""Reusable TimeLens inference runner for temporal search subclips."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
import os
import re
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from evaluation.timelens_search import Interval
from evaluation.utils import GROUNDER_PROMPT, GROUNDER_PROMPT_TEXT_TIMESTAMP
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


SUBCLIP_PROMPT_PREFIX = (
    "You are analyzing the portion from {clip_start:.3f} to {clip_end:.3f} "
    "seconds in the original video. Return the event start and end times in "
    "original video seconds. "
)


@dataclass
class RunnerPrediction:
    answer: str
    timestamps: List[Any]
    error: Optional[str] = None


@dataclass
class RunnerVerdict:
    answer: str
    choice: str
    confidence: float
    failure_tag: str = "uncertain"
    error: Optional[str] = None


def format_gib(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if float(value).is_integer():
        return f"{int(value)}GiB"
    return f"{value}GiB"


def resolve_device_settings(args: Any) -> Any:
    capability = None
    device_name = "cpu"
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        device_name = torch.cuda.get_device_name(0)

    dtype_arg = getattr(args, "dtype", "auto")
    if dtype_arg == "auto":
        if capability is None:
            dtype = torch.float32
        elif capability[0] >= 8:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = DTYPE_MAP[dtype_arg]

    attn_arg = getattr(args, "attn_implementation", "auto")
    if attn_arg == "auto":
        if capability is not None and capability[0] >= 8:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
    else:
        attn_implementation = attn_arg

    print(
        "Resolved runtime config: "
        f"device={device_name}, capability={capability}, "
        f"dtype={dtype}, attn_implementation={attn_implementation}"
    )
    return dtype, attn_implementation


def repeat_kv_to_match_heads(hidden_states: torch.Tensor, target_heads: int) -> torch.Tensor:
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
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        print(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`; "
            "falling back may be required."
        )

    sdpa_kwargs: Dict[str, Any] = {}
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


def resolve_model_parallel_args(args: Any) -> Dict[str, Any]:
    if getattr(args, "tp_plan", "none") != "none":
        return {"tp_plan": args.tp_plan}

    model_parallel_args: Dict[str, Any] = {
        "device_map": getattr(args, "device", "auto"),
    }
    max_memory: Dict[Any, str] = {}
    max_gpu_memory_gib = getattr(args, "max_gpu_memory_gib", None)
    if max_gpu_memory_gib is not None and torch.cuda.is_available():
        gpu_budget = format_gib(max_gpu_memory_gib)
        for gpu_idx in range(torch.cuda.device_count()):
            max_memory[gpu_idx] = gpu_budget
    cpu_offload_gib = getattr(args, "cpu_offload_gib", None)
    if cpu_offload_gib is not None:
        max_memory["cpu"] = format_gib(cpu_offload_gib)
    if max_memory:
        model_parallel_args["max_memory"] = max_memory

    offload_folder = getattr(args, "offload_folder", None)
    if offload_folder:
        os.makedirs(offload_folder, exist_ok=True)
        model_parallel_args["offload_folder"] = offload_folder

    offload_state_dict = getattr(args, "offload_state_dict", "auto")
    if offload_state_dict != "auto":
        model_parallel_args["offload_state_dict"] = offload_state_dict == "true"
    return model_parallel_args


def normalize_model_device(device: Any) -> Optional[torch.device]:
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


def resolve_input_device(model: Any) -> torch.device:
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


def summarize_device_map(model: Any) -> None:
    hf_device_map = getattr(model, "hf_device_map", None)
    if not hf_device_map:
        return
    summary: Dict[str, int] = {}
    for device_value in hf_device_map.values():
        key = str(device_value)
        summary[key] = summary.get(key, 0) + 1
    print(f"HF device_map summary: {summary}")


class TimeLensRunner:
    """Load TimeLens once and run deterministic grounding on requested windows."""

    def __init__(self, args: Any):
        self.args = args
        ALL_ATTENTION_FUNCTIONS["sdpa"] = patched_sdpa_attention_forward

        self.dtype, self.attn_implementation = resolve_device_settings(args)
        self.model_parallel_args = resolve_model_parallel_args(args)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=self.dtype,
            attn_implementation=self.attn_implementation,
            low_cpu_mem_usage=True,
            **self.model_parallel_args,
        ).eval()
        summarize_device_map(self.model)
        self.input_device = resolve_input_device(self.model)
        print(f"Resolved model input device: {self.input_device}")

        self.processor = AutoProcessor.from_pretrained(
            args.model_path,
            padding_side="left",
            do_resize=False,
            use_fast=False,
            trust_remote_code=True,
        )

    def _downsample_rate(self) -> int:
        model_path = self.args.model_path.lower()
        if "qwen3" in model_path or "timelens-8b" in model_path:
            return 32
        if "qwen2" in model_path or "timelens-7b" in model_path:
            return 28
        raise NotImplementedError(f"Model {self.args.model_path} not supported yet.")

    def _prompt(self, query: str, clip: Interval) -> str:
        model_path = self.args.model_path.lower()
        if "timelens-7b" in model_path:
            base_prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP
        else:
            base_prompt = GROUNDER_PROMPT
        prefix = SUBCLIP_PROMPT_PREFIX.format(
            clip_start=clip.start,
            clip_end=clip.end,
        )
        return prefix + base_prompt.format(query)

    def _build_inputs(
        self,
        anno: Dict[str, Any],
        clip: Interval,
        fps_override: Optional[int] = None,
        prompt_override: Optional[str] = None,
    ) -> Any:
        downsample_rate = self._downsample_rate()
        fps = fps_override
        if fps is None:
            fps = getattr(self.args, "zoom_fps", getattr(self.args, "fps", 4))
        min_tokens = getattr(self.args, "min_tokens", 64)
        total_tokens = getattr(self.args, "total_tokens", 14336)
        video_item = {
            "type": "video",
            "video": anno["video_path"],
            "min_pixels": min_tokens * downsample_rate * downsample_rate,
            "total_pixels": total_tokens * downsample_rate * downsample_rate,
            "fps": fps,
            "video_start": float(clip.start),
            "video_end": float(clip.end),
        }
        messages = [
            {
                "role": "user",
                "content": [
                    video_item,
                    {
                        "type": "text",
                        "text": prompt_override
                        if prompt_override is not None
                        else self._prompt(anno["query"], clip),
                    },
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_path = self.args.model_path.lower()
        if "timelens-7b" in model_path:
            images, videos = process_vision_info(messages, return_video_metadata=True)
            return self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        if "qwen3" in model_path or "timelens-8b" in model_path:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            return self.processor(
                text=[text],
                images=images,
                videos=list(videos),
                video_metadata=list(video_metadatas),
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        if "qwen2" in model_path:
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            return self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        raise NotImplementedError(f"Model {self.args.model_path} not supported yet.")

    def _generate_answer(
        self,
        anno: Dict[str, Any],
        clip: Interval,
        fps_override: Optional[int] = None,
        prompt_override: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> str:
        inputs = None
        output_ids = None
        generated_ids_trimmed = None
        try:
            inputs = self._build_inputs(
                anno,
                clip,
                fps_override=fps_override,
                prompt_override=prompt_override,
            )
            non_blocking = self.input_device.type == "cuda"
            inputs = inputs.to(self.input_device, non_blocking=non_blocking)

            generate_kwargs = dict(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=(
                    max_new_tokens_override
                    if max_new_tokens_override is not None
                    else getattr(self.args, "max_new_tokens", 512)
                ),
            )
            if getattr(self.args, "use_cache", "auto") != "auto":
                generate_kwargs["use_cache"] = self.args.use_cache == "true"

            if (
                getattr(self.args, "attn_implementation", "auto") == "sdpa"
                and getattr(self.args, "sdpa_backend", "auto") != "auto"
            ):
                generate_context = sdpa_kernel([SDPA_BACKEND_MAP[self.args.sdpa_backend]])
            else:
                generate_context = nullcontext()

            with generate_context:
                output_ids = self.model.generate(**generate_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, output_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return answer
        finally:
            del output_ids
            del generated_ids_trimmed
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def predict_interval(
        self,
        anno: Dict[str, Any],
        clip: Interval,
        fps_override: Optional[int] = None,
    ) -> RunnerPrediction:
        try:
            answer = self._generate_answer(
                anno,
                clip,
                fps_override=fps_override,
            )
            timestamps = extract_time(answer)
            if timestamps:
                print(
                    f"TimeLens subclip {clip.start:.3f}-{clip.end:.3f}: "
                    f"timestamps={timestamps}"
                )
            else:
                print(
                    f"TimeLens subclip {clip.start:.3f}-{clip.end:.3f}: "
                    f"no timestamps extracted; answer={answer}"
                )
            return RunnerPrediction(answer=answer, timestamps=timestamps)
        except Exception as exc:
            if getattr(self.args, "skip_video_errors", "false") != "true":
                raise
            return RunnerPrediction(
                answer=f"SKIPPED_DUE_TO_VIDEO_ERROR: {exc}",
                timestamps=[],
                error=str(exc),
            )

    @staticmethod
    def _parse_verdict(answer: str) -> RunnerVerdict:
        payload: Dict[str, Any] = {}
        for match in reversed(re.findall(r"\{.*?\}", answer, flags=re.DOTALL)):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break

        choice = str(payload.get("choice", "")).strip().upper()
        choice_map = {
            "A": "A",
            "B": "B",
            "CANDIDATE_A": "A",
            "CANDIDATE A": "A",
            "CANDIDATE_B": "B",
            "CANDIDATE B": "B",
            "UNCERTAIN": "UNCERTAIN",
            "TIE": "UNCERTAIN",
        }
        choice = choice_map.get(choice, "")
        if not choice:
            match = re.search(
                r"(?:choice|better)\s*[:=]\s*[\"']?(A|B|UNCERTAIN|TIE)\b",
                answer,
                flags=re.IGNORECASE,
            )
            choice = choice_map.get(match.group(1).upper(), "UNCERTAIN") if match else "UNCERTAIN"

        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence <= 0.0:
            match = re.search(
                r"confidence\s*[:=]\s*[\"']?([01](?:\.\d+)?)",
                answer,
                flags=re.IGNORECASE,
            )
            if match:
                confidence = float(match.group(1))
        confidence = min(max(confidence, 0.0), 1.0)
        failure_tag = str(payload.get("failure_tag", "uncertain"))
        return RunnerVerdict(
            answer=answer,
            choice=choice,
            confidence=confidence,
            failure_tag=failure_tag,
        )

    def verify_pair(
        self,
        anno: Dict[str, Any],
        candidate_a: Interval,
        candidate_b: Interval,
        fps_override: Optional[int] = None,
    ) -> RunnerVerdict:
        duration = float(anno["duration"])
        context_margin = max(
            2.0,
            0.25 * max(candidate_a.length, candidate_b.length),
        )
        clip = Interval(
            min(candidate_a.start, candidate_b.start) - context_margin,
            max(candidate_a.end, candidate_b.end) + context_margin,
        ).clamp(duration)
        prompt = (
            "You are a blinded temporal-grounding verifier. The original query is:\n"
            f"{anno['query']}\n\n"
            f"Candidate A: [{candidate_a.start:.3f}, {candidate_a.end:.3f}] seconds.\n"
            f"Candidate B: [{candidate_b.start:.3f}, {candidate_b.end:.3f}] seconds.\n"
            f"The visible context covers [{clip.start:.3f}, {clip.end:.3f}] seconds in the original video.\n"
            "Judge which candidate more tightly contains the complete queried event. Check visible "
            "evidence immediately before the start, inside the interval, and immediately after the end. "
            "Do not trust either candidate by default. If the evidence is insufficient or they are "
            "effectively tied, choose UNCERTAIN. Return exactly one JSON object with no markdown:\n"
            '{"choice":"A|B|UNCERTAIN","confidence":0.0,'
            '"failure_tag":"too_early|too_late|too_long|too_short|wrong_event|uncertain",'
            '"reason":"brief evidence-based reason"}'
        )
        try:
            answer = self._generate_answer(
                anno,
                clip,
                fps_override=fps_override,
                prompt_override=prompt,
                max_new_tokens_override=min(
                    getattr(self.args, "max_new_tokens", 512),
                    192,
                ),
            )
            verdict = self._parse_verdict(answer)
            print(
                f"TimeLens critic A={candidate_a.as_list()} B={candidate_b.as_list()}: "
                f"choice={verdict.choice} confidence={verdict.confidence:.3f} "
                f"failure_tag={verdict.failure_tag}"
            )
            return verdict
        except Exception as exc:
            if getattr(self.args, "skip_video_errors", "false") != "true":
                raise
            return RunnerVerdict(
                answer=f"SKIPPED_DUE_TO_VIDEO_ERROR: {exc}",
                choice="UNCERTAIN",
                confidence=0.0,
                failure_tag="video_error",
                error=str(exc),
            )
