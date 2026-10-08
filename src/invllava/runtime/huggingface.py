"""Small, frozen loading boundaries for upstream Hugging Face references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def require_immutable_revision(revision: str) -> None:
    if revision in {"", "main", "pending-freeze"}:
        raise ValueError("reference loading requires an immutable model revision")


@dataclass(frozen=True)
class HuggingFaceLLaVARuntime:
    model: Any
    processor: Any
    checkpoint: str
    revision: str


def load_hf_llava_runtime(
    checkpoint: str,
    *,
    revision: str,
    dtype: torch.dtype,
    device: str,
    attention_backend: str | None = None,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> HuggingFaceLLaVARuntime:
    """Load one verified LLaVA reference for inference or analysis."""

    require_immutable_revision(revision)
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    shared = {
        "revision": revision,
        "local_files_only": local_files_only,
        "token": token,
    }
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        trust_remote_code=False,
        **shared,
    )
    model_kwargs: dict[str, Any] = {
        **shared,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attention_backend is not None:
        model_kwargs["attn_implementation"] = attention_backend
    model = LlavaForConditionalGeneration.from_pretrained(checkpoint, **model_kwargs)
    model.to(device).eval()
    return HuggingFaceLLaVARuntime(
        model=model,
        processor=processor,
        checkpoint=checkpoint,
        revision=revision,
    )
