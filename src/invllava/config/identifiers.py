"""Stable identifiers for scientific and protocol state."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from pydantic import BaseModel


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_id(value: Any, *, prefix: str, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:length]}"


def _protocol_source_payload(source: Any) -> dict[str, Any]:
    data = source.model_dump(mode="json") if isinstance(source, BaseModel) else dict(source)
    # A checksum names downloaded bytes independently of their mirror. For
    # repositories or manual inputs without a whole-source checksum, retain the
    # locator and source kind as part of the scientific identity.
    if data.get("sha256"):
        return {
            "sha256": data["sha256"],
            "subset": data.get("subset"),
        }
    return {
        "id": data["id"],
        "kind": data.get("kind"),
        "location": data.get("location"),
        "revision": data.get("revision"),
        "repo_type": data.get("repo_type"),
        "subset": data.get("subset"),
    }


def benchmark_protocol_payload(spec: Any) -> dict[str, Any]:
    """Return the stable, scientifically relevant identity of a benchmark."""

    generation = spec.generation.model_dump(mode="json")
    return {
        "schema": "invllava-benchmark-protocol-v1",
        "id": spec.id,
        "split": spec.split,
        "protocol_revision": spec.protocol_revision,
        "task_type": spec.task_type,
        "annotations": _protocol_source_payload(spec.annotations),
        "images": _protocol_source_payload(spec.images) if spec.images is not None else None,
        "protocol_sources": sorted(
            (_protocol_source_payload(source) for source in spec.protocol_sources),
            key=canonical_json,
        ),
        "conversation_template": spec.conversation_template,
        "prompt_template": spec.prompt_template,
        "answer_extraction": spec.answer_extraction,
        "scorer": spec.scorer,
        "external_only": spec.external_only,
        "generation": generation,
    }


def benchmark_protocol_id(spec: Any) -> str:
    return content_id(benchmark_protocol_payload(spec), prefix="protocol")


def scientific_payload(resolved: Any) -> dict[str, Any]:
    data = (
        resolved.model_dump(mode="json")
        if isinstance(resolved, BaseModel)
        else deepcopy(dict(resolved))
    )
    # Hardware and paths affect reproducibility records, but do not define the
    # scientific experiment identity.  Process count and accumulation are two
    # execution schedules for the same fixed per-device microbatch and optimizer
    # update batch, so normalize them into the effective global batch.
    runtime = data.get("runtime", {})
    training = data.get("training", {})
    if runtime and training:
        training["effective_global_batch_size"] = (
            training["per_device_batch_size"]
            * training["gradient_accumulation_steps"]
            * runtime["num_processes"]
        )
        training.pop("gradient_accumulation_steps", None)
    data.pop("runtime", None)
    data.pop("source_files", None)
    return data
