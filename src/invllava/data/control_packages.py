"""Seal selected training controls with hard-linked images and parent evidence."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.data.audit import audit_image_integrity, audit_samples
from invllava.data.manifest import build_prepared_manifest
from invllava.data.prepare import write_normalized_jsonl
from invllava.data.types import ConversationSample


def _link_images(samples: list[ConversationSample], package: Path) -> list[ConversationSample]:
    linked: dict[Path, Path] = {}
    result: list[ConversationSample] = []
    for sample in samples:
        images: list[Path] = []
        for source in sample.images:
            resolved = source.resolve(strict=True)
            target = linked.get(resolved)
            if target is None:
                identity = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:20]
                suffix = resolved.suffix.lower() or ".image"
                target = package / "images" / f"{identity}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(resolved, target)
                except OSError as error:
                    raise OSError(
                        "control packages require source data and output on one filesystem "
                        "so images can be hard-linked without payload copies"
                    ) from error
                linked[resolved] = target
            images.append(target)
        result.append(replace(sample, images=tuple(images)))
    return result


def materialize_control_package(
    root: Path,
    *,
    name: str,
    data_id: str,
    revision: str,
    samples: list[ConversationSample],
    parents: dict[str, str],
    token_counts: dict[str, int],
    workers: int,
) -> dict[str, object]:
    """Atomically publish a relocatable package; never copy image payloads."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("control package name must be one safe path component")
    if not samples or workers <= 0 or not revision:
        raise ValueError("samples, positive workers, and a revision are required")
    if any(token_counts.get(sample.id, 0) <= 0 for sample in samples):
        raise ValueError("every selected sample needs a positive surviving-target count")
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("control sample identities must be unique")
    destination = root / name
    if destination.exists():
        raise FileExistsError(destination)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        linked = _link_images(samples, temporary)
        normalized = temporary / "train.jsonl"
        write_normalized_jsonl(linked, normalized)
        descriptor = temporary / "selection.json"
        selected_tokens = sum(token_counts[sample.id] for sample in samples)
        atomic_write_json(
            descriptor,
            {
                "schema_version": 1,
                "algorithm": revision,
                "condition": name,
                "parents": parents,
                "sample_ids": [sample.id for sample in samples],
                "samples": len(samples),
                "supervised_tokens": selected_tokens,
            },
        )
        image_audit = audit_image_integrity(linked, image_root=temporary, workers=workers)
        if not image_audit.to_dict()["passed"]:
            raise ValueError("control package image integrity audit failed")
        image_evidence = {
            **image_audit.to_dict(),
            "annotation_sha256": sha256_file(descriptor),
            "source_revision": revision,
            "image_root": str(destination),
            "selected_sources": None,
        }
        manifest = build_prepared_manifest(
            data_id=data_id,
            source_revision=revision,
            source_path=descriptor,
            normalized_path=normalized,
            samples=linked,
            audit=audit_samples(linked),
            image_integrity=image_evidence,
        )
        manifest = replace(manifest, normalized_path=str(destination / "train.jsonl"))
        manifest.write(temporary / "train.manifest.json")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "condition": name,
        "data_id": data_id,
        "path": str(destination),
        "samples": len(samples),
        "supervised_tokens": selected_tokens,
        "manifest_sha256": sha256_file(destination / "train.manifest.json"),
    }
