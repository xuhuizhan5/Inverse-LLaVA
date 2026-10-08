"""Pinned, streaming materialization of the paper-era LLaVA MME protocol."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import (
    MME_COGNITION_CATEGORIES,
    MME_PERCEPTION_CATEGORIES,
    canonicalize_mme_question,
    write_examples,
)
from invllava.eval.prompts import render_benchmark_prompt
from invllava.eval.types import EvaluationExample

_LLAVA_QUESTION_MEMBER = "MME/llava_mme.jsonl"
_EXPECTED_QUESTIONS = {"perception": 2_114, "cognition": 260}
_EXPECTED_IMAGES = {"perception": 1_057, "cognition": 130}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _load_released_questions(archive: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(_LLAVA_QUESTION_MEMBER)
        except KeyError as error:
            raise ValueError("LLaVA evaluation archive lacks its MME question fixture") from error
        if info.file_size > 16 * 1024**2:
            raise ValueError("LLaVA MME question fixture exceeds the 16 MiB safety bound")
        rows = [json.loads(line) for line in bundle.read(info).decode("utf-8").splitlines()]
    if len(rows) != 2_374:
        raise ValueError("released LLaVA MME fixture must contain 2,374 questions")
    return rows


def _member_index(bundle: zipfile.ZipFile) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    annotations: dict[str, str] = {}
    categories = set(MME_PERCEPTION_CATEGORIES + MME_COGNITION_CATEGORIES)
    for info in bundle.infolist():
        if info.is_dir():
            continue
        parts = PurePosixPath(info.filename).parts
        category_positions = [index for index, value in enumerate(parts) if value in categories]
        if len(category_positions) != 1:
            continue
        category = parts[category_positions[0]]
        name = parts[-1]
        suffix = PurePosixPath(name).suffix.lower()
        key = f"{category}/{name}"
        target = images if suffix in _IMAGE_SUFFIXES else annotations if suffix == ".txt" else None
        if target is None:
            continue
        if key in target:
            raise ValueError(f"MME archive contains duplicate member key {key}")
        target[key] = info.filename
    if len(images) != 1_187 or len(annotations) != 1_187:
        raise ValueError("MME archive must contain 1,187 images and 1,187 paired annotation files")
    return images, annotations


def _annotation_pairs(bundle: zipfile.ZipFile, member: str) -> tuple[tuple[str, str], ...]:
    lines = bundle.read(member).decode("utf-8").splitlines()
    if len(lines) != 2:
        raise ValueError(f"MME annotation {member} must contain exactly two questions")
    pairs = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split("\t")
        if len(fields) != 2:
            raise ValueError(f"MME annotation {member}:{line_number} is malformed")
        question, answer = fields
        if answer not in {"Yes", "No"}:
            raise ValueError(f"MME annotation {member}:{line_number} has a non-binary answer")
        pairs.append((question, answer))
    return tuple(pairs)


def _save_archive_image(
    bundle: zipfile.ZipFile,
    member: str,
    *,
    group_id: str,
    physical_root: Path,
    logical_root: Path,
) -> tuple[Path, dict[str, Any]]:
    payload = bundle.read(member)
    suffix = PurePosixPath(member).suffix.lower()
    name = hashlib.sha256(group_id.encode()).hexdigest() + suffix
    physical = physical_root / name
    with Image.open(BytesIO(payload)) as image:
        image.load()
        image_format = str(image.format or "unknown")
        pixel_sha256 = canonical_rgb_sha256(image.convert("RGB"))
    atomic_write_bytes(physical, payload)
    return logical_root / name, {
        "file": f"images/{name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": pixel_sha256,
        "size_bytes": len(payload),
        "format": image_format,
        "source_member": member,
    }


def materialize_mme_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Download two checksummed archives and materialize one MME domain."""

    domains = {"mme-perception": "perception", "mme-cognition": "cognition"}
    if spec.id not in domains:
        raise ValueError(f"{spec.id} is not an MME domain benchmark")
    if spec.annotations.kind != "http" or not spec.annotations.sha256:
        raise ValueError("MME requires the checksummed LLaVA evaluation archive")
    if spec.images is None or spec.images.kind != "http" or not spec.images.sha256:
        raise ValueError("MME requires a checksummed image/annotation archive")
    if not spec.images.revision or spec.images.revision == "pending-freeze":
        raise ValueError("MME image archive revision must be immutable")

    cache_root = Path(cache_dir)
    prompt_archive = download_http(
        spec.annotations.location,
        cache_root / "protocols" / spec.annotations.id / "eval.zip",
        sha256=spec.annotations.sha256,
    )
    data_archive = download_http(
        spec.images.location,
        cache_root / "benchmarks" / spec.images.id / "MME_Benchmark_release_version.zip",
        sha256=spec.images.sha256,
    )
    released = _load_released_questions(prompt_archive)
    domain = domains[spec.id]
    domain_categories = set(
        MME_PERCEPTION_CATEGORIES if domain == "perception" else MME_COGNITION_CATEGORIES
    )
    selected = [row for row in released if str(row.get("category")) in domain_categories]
    if len(selected) != _EXPECTED_QUESTIONS[domain]:
        raise ValueError(f"released LLaVA MME {domain} fixture has an unexpected size")

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    physical_images = temporary / "images"
    physical_images.mkdir()
    logical_images = Path("images")
    try:
        examples: list[EvaluationExample] = []
        image_records: dict[str, dict[str, Any]] = {}
        pair_offsets: Counter[str] = Counter()
        category_counts: Counter[str] = Counter()
        with zipfile.ZipFile(data_archive) as bundle:
            image_members, annotation_members = _member_index(bundle)
            annotation_cache: dict[str, tuple[tuple[str, str], ...]] = {}
            image_paths: dict[str, Path] = {}
            for position, row in enumerate(selected, start=1):
                category = str(row["category"])
                group_id = str(row["question_id"])
                released_image = str(row["image"])
                if released_image != group_id or not released_image.startswith(f"{category}/"):
                    raise ValueError(f"released MME fixture has inconsistent image ID {group_id}")
                pair_index = pair_offsets[group_id]
                pair_offsets[group_id] += 1
                if pair_index >= 2:
                    raise ValueError(f"released MME fixture has more than two rows for {group_id}")

                annotation_key = f"{category}/{PurePosixPath(group_id).with_suffix('.txt').name}"
                if annotation_key not in annotation_members:
                    raise ValueError(f"MME archive lacks annotation {annotation_key}")
                pairs = annotation_cache.setdefault(
                    group_id,
                    _annotation_pairs(bundle, annotation_members[annotation_key]),
                )
                question, answer = pairs[pair_index]
                canonical_question = canonicalize_mme_question(question)
                if str(row["text"]) != canonical_question:
                    raise ValueError(
                        f"MME prompt differs from LLaVA fixture for {group_id}/{pair_index}"
                    )

                if group_id not in image_paths:
                    image_key = f"{category}/{PurePosixPath(released_image).name}"
                    if image_key not in image_members:
                        raise ValueError(f"MME archive lacks image {released_image}")
                    image_path, image_record = _save_archive_image(
                        bundle,
                        image_members[image_key],
                        group_id=group_id,
                        physical_root=physical_images,
                        logical_root=logical_images,
                    )
                    image_paths[group_id] = image_path
                    image_records[str(image_record["file"])] = image_record
                examples.append(
                    EvaluationExample(
                        id=f"{group_id}/{pair_index}",
                        prompt=render_benchmark_prompt(spec, {"question": canonical_question}),
                        images=(image_paths[group_id],),
                        references=(answer,),
                        group_id=group_id,
                        metadata={"category": category, "domain": domain},
                    )
                )
                category_counts[category] += 1
                if progress is not None and (position % 250 == 0 or position == len(selected)):
                    progress("convert", position, len(selected))

        invalid_groups = [group for group, count in pair_offsets.items() if count != 2]
        if invalid_groups or len(pair_offsets) != _EXPECTED_IMAGES[domain]:
            raise ValueError("MME fixture does not contain exactly two questions per image")
        if set(category_counts) != domain_categories:
            raise ValueError(f"MME {domain} fixture does not cover every declared category")

        examples_path = temporary / "examples.jsonl"
        write_examples(examples, examples_path)
        image_audit = audit_image_integrity(
            (
                ImageReferenceSet((physical_images / example.images[0].name,), spec.id)
                for example in examples
            ),
            image_root=physical_images,
            workers=8,
            progress_every=250,
            progress=(
                (lambda completed, total: progress("image-audit", completed, total))
                if progress is not None
                else None
            ),
        )
        image_audit_payload = {
            "schema_version": 1,
            "benchmark_id": spec.id,
            "protocol_revision": spec.protocol_revision,
            "image_root": "images",
            **image_audit.to_dict(),
        }
        image_audit_path = temporary / "image-integrity.json"
        atomic_write_json(image_audit_path, image_audit_payload)
        if image_audit.unique_images != _EXPECTED_IMAGES[domain] or not image_audit.passed:
            raise RuntimeError("materialized MME image integrity audit failed")
        atomic_write_json(
            temporary / "manifest.json",
            {
                "format": "invllava-eval-dataset-v1",
                "invllava_version": __version__,
                "execution_source_sha256": optional_sha256_environment(
                    "INVLLAVA_EXECUTION_SOURCE_SHA256"
                ),
                "benchmark_id": spec.id,
                "protocol_revision": spec.protocol_revision,
                "protocol_config_sha256": config_sha256,
                "conversation_template": spec.conversation_template,
                "dataset": spec.images.location,
                "dataset_revision": spec.images.revision,
                "dataset_sha256": spec.images.sha256,
                "annotation_source": spec.annotations.location,
                "annotation_sha256": spec.annotations.sha256,
                "released_fixture_member": _LLAVA_QUESTION_MEMBER,
                "split": spec.split,
                "domain": domain,
                "sample_count": len(examples),
                "category_counts": dict(sorted(category_counts.items())),
                "examples_sha256": sha256_file(examples_path),
                "images": [image_records[key] for key in sorted(image_records)],
                "image_integrity": {
                    **image_audit_payload,
                    "report_sha256": sha256_file(image_audit_path),
                },
            },
        )
        temporary.chmod(0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
