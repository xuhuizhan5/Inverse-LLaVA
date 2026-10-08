from __future__ import annotations

import re

from invllava.config.schema import ResolvedExperiment

_SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HUGGINGFACE_COMMIT = re.compile(r"[0-9a-f]{40}")


def validate_execution_contract(config: ResolvedExperiment) -> list[str]:
    """Return actionable warnings after strict schema validation succeeds."""

    warnings: list[str] = []
    effective_batch = (
        config.training.per_device_batch_size
        * config.training.gradient_accumulation_steps
        * config.runtime.num_processes
    )
    if effective_batch < 32:
        warnings.append(f"effective global batch is only {effective_batch}")
    if config.runtime.accelerator == "cpu" and config.model.torch_dtype != "float32":
        warnings.append("CPU validation should override model.torch_dtype to float32")
    if config.model.architecture == "inverse_llava" and config.model.vision.freeze is False:
        warnings.append("vision encoder is trainable; this is not the canonical manuscript setting")
    return warnings


def require_frozen_execution(config: ResolvedExperiment) -> None:
    pending: list[str] = []
    if config.model.language.revision == "pending-freeze":
        pending.append("model.language.revision")
    if config.model.vision.revision == "pending-freeze":
        pending.append("model.vision.revision")
    if config.model.fusion.initialization_id.startswith("pending-"):
        pending.append("model.fusion.initialization_id")
    if config.initial_checkpoint_id is not None and not _SHA256_ID.fullmatch(
        config.initial_checkpoint_id
    ):
        pending.append("initial_checkpoint_id (expected sha256:<64 lowercase hex>)")
    if (
        config.model.projector is not None
        and config.model.projector.initialization == "checkpoint"
        and config.model.projector.initial_checkpoint_id != "embedded-in-official-checkpoint"
        and not _SHA256_ID.fullmatch(config.model.projector.initial_checkpoint_id or "")
    ):
        pending.append("model.projector.initial_checkpoint_id (expected sha256:<64 lowercase hex>)")
    sources = (config.data.annotation, *config.data.image_sources)
    for source in sources:
        if not source.required or source.kind == "generated":
            continue
        if source.kind == "huggingface":
            if not source.revision or not _HUGGINGFACE_COMMIT.fullmatch(source.revision):
                pending.append(f"data source {source.id} immutable Hugging Face commit")
            if source.sha256 is not None and not _SHA256.fullmatch(source.sha256):
                pending.append(f"data source {source.id} valid optional sha256")
        elif not source.sha256 or not _SHA256.fullmatch(source.sha256):
            pending.append(f"data source {source.id} sha256")
    if pending:
        raise ValueError("execution config contains unresolved provenance: " + ", ".join(pending))
