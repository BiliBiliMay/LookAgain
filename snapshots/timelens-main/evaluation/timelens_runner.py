"""Reusable TimeLens inference runner for ActiveZoom subclips."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
from typing import Any, Dict, List, Optional

import torch
from qwen_vl_utils import process_vision_info
from torch.nn.attention import sdpa_kernel
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from evaluation.activezoom import Interval
from evaluation.eval_dataloader import (
    SDPA_BACKEND_MAP,
    patched_sdpa_attention_forward,
    resolve_device_settings,
    resolve_input_device,
    resolve_model_parallel_args,
    summarize_device_map,
)
from evaluation.utils import GROUNDER_PROMPT, GROUNDER_PROMPT_TEXT_TIMESTAMP
from timelens.utils import extract_time


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

    def _build_inputs(self, anno: Dict[str, Any], clip: Interval) -> Any:
        downsample_rate = self._downsample_rate()
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
                    {"type": "text", "text": self._prompt(anno["query"], clip)},
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

    def predict_interval(self, anno: Dict[str, Any], clip: Interval) -> RunnerPrediction:
        inputs = None
        output_ids = None
        generated_ids_trimmed = None
        try:
            inputs = self._build_inputs(anno, clip)
            non_blocking = self.input_device.type == "cuda"
            inputs = inputs.to(self.input_device, non_blocking=non_blocking)

            generate_kwargs = dict(
                **inputs,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=512,
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
            timestamps = extract_time(answer)
            if timestamps:
                print(
                    f"ActiveZoom subclip {clip.start:.3f}-{clip.end:.3f}: "
                    f"timestamps={timestamps}"
                )
            else:
                print(
                    f"ActiveZoom subclip {clip.start:.3f}-{clip.end:.3f}: "
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
        finally:
            del output_ids
            del generated_ids_trimmed
            del inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
