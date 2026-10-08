"""One composition root for checkpoint-backed native inference tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from invllava.artifacts.hashing import sha256_file
from invllava.config.loader import ConfigRepository
from invllava.config.schema import ResolvedExperiment
from invllava.config.validation import require_frozen_execution
from invllava.eval.runtime import InverseGenerator
from invllava.model.loaders import LoadReport, build_model, load_image_processor, load_tokenizer
from invllava.model.modeling import (
    InverseLLaVAForConditionalGeneration,
    LLaVAReferenceForConditionalGeneration,
)
from invllava.runtime.cache import configure_runtime_cache
from invllava.runtime.numerics import configure_torch_numerics
from invllava.runtime.optimization import KernelOptimizationReport, configure_model_kernels
from invllava.train.checkpoint import MODEL_DELTA_FILENAME, load_trainable_weights


@dataclass(frozen=True)
class NativeInferenceRuntime:
    """Fully loaded native model and the evidence needed by artifact writers."""

    resolved: ResolvedExperiment
    model: InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration
    tokenizer: Any
    image_processor: Any
    generator: InverseGenerator
    base_load_report: LoadReport
    kernel_report: KernelOptimizationReport
    numerical_policy: dict[str, Any]
    checkpoint_sha256: str

    @property
    def checkpoint_id(self) -> str:
        return f"ckpt-{self.checkpoint_sha256[:16]}"


def load_native_inference_runtime(
    *,
    experiment: str | Path,
    checkpoint: str | Path,
    config_root: str | Path,
    runtime_ref: str | None,
    device: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    require_portable: bool = False,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> NativeInferenceRuntime:
    """Resolve, verify, and load the shared native inference stack exactly once."""

    resolved = ConfigRepository(config_root).resolve(experiment, runtime_ref=runtime_ref)
    require_frozen_execution(resolved)
    if require_portable and resolved.runtime.kernel_optimization.compile_scope != "none":
        raise ValueError(
            "this analysis requires the portable uncompiled runtime; select a runtime "
            "with kernel_optimization.compile_scope=none"
        )
    configure_runtime_cache(resolved.runtime.cache_root)
    numerical_policy = configure_torch_numerics(resolved.runtime)

    model, load_report = build_model(
        resolved.model,
        attention_backend=resolved.runtime.attention_backend,
        local_files_only=local_files_only,
        token=token,
    )
    load_trainable_weights(checkpoint, model)
    kernel_report = configure_model_kernels(model, resolved.runtime.kernel_optimization)
    model.to(device).eval()

    tokenizer = load_tokenizer(
        resolved.model,
        local_files_only=local_files_only,
        token=token,
    )
    image_processor = load_image_processor(
        resolved.model.vision,
        local_files_only=local_files_only,
        token=token,
    )
    generator = InverseGenerator(
        model,
        tokenizer,
        image_processor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_to_square=resolved.model.vision.aspect_ratio == "pad",
    )
    delta = Path(checkpoint) / MODEL_DELTA_FILENAME
    return NativeInferenceRuntime(
        resolved=resolved,
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        generator=generator,
        base_load_report=load_report,
        kernel_report=kernel_report,
        numerical_policy=numerical_policy,
        checkpoint_sha256=sha256_file(delta),
    )
