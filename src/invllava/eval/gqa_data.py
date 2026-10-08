"""Checksummed, relocatable materialization of GQA test-dev balanced."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.schema import BenchmarkSpec, DataSource
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import write_examples
from invllava.eval.prompts import render_benchmark_prompt
from invllava.eval.protocols.gqa import normalize_gqa_prediction
from invllava.eval.types import EvaluationExample

_EXPECTED_ROWS = 12_578
_QUESTION_SUFFIX = "testdev_balanced_questions.json"
_LLAVA_QUESTION_MEMBER = "gqa/llava_gqa_testdev_balanced.jsonl"
_LLAVA_SUFFIX = "\nAnswer the question using a single word or phrase."


def _protocol_source(spec: BenchmarkSpec, source_id: str) -> DataSource:
    matches = [source for source in spec.protocol_sources if source.id == source_id]
    if len(matches) != 1:
        raise ValueError(f"GQA requires one protocol source named {source_id}")
    source = matches[0]
    if source.kind != "http" or not source.sha256:
        raise ValueError(f"GQA protocol source {source_id} must be checksummed HTTP")
    return source


def _unique_member(bundle: zipfile.ZipFile, *, suffix: str) -> zipfile.ZipInfo:
    matches = [item for item in bundle.infolist() if item.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one member ending with {suffix}")
    return matches[0]


def _load_json_member(bundle: zipfile.ZipFile, info: zipfile.ZipInfo) -> Any:
    if info.file_size > 128 * 1024**2:
        raise ValueError(f"JSON member exceeds the 128 MiB safety bound: {info.filename}")
    with bundle.open(info) as stream:
        return json.load(stream)


def _load_jsonl_member(bundle: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    try:
        info = bundle.getinfo(member)
    except KeyError as error:
        raise ValueError(f"archive lacks required member: {member}") from error
    if info.file_size > 64 * 1024**2:
        raise ValueError(f"JSONL member exceeds the 64 MiB safety bound: {member}")
    rows: list[dict[str, Any]] = []
    with bundle.open(info) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"GQA fixture row {line_number} is not an object")
            rows.append(value)
    return rows


def build_gqa_examples(
    questions: dict[str, Any],
    llava_rows: list[dict[str, Any]],
    *,
    spec: BenchmarkSpec,
    logical_image_root: str | Path = "images",
) -> list[EvaluationExample]:
    """Join official answers to LLaVA's released prompt and image ordering."""

    if not isinstance(questions, dict) or not questions:
        raise ValueError("official GQA questions are empty or malformed")
    root = Path(logical_image_root)
    examples: list[EvaluationExample] = []
    seen: set[str] = set()
    for position, row in enumerate(llava_rows):
        sample_id = str(row.get("question_id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate LLaVA GQA question ID at row {position}")
        seen.add(sample_id)
        source = questions.get(sample_id)
        if not isinstance(source, dict):
            raise ValueError(f"LLaVA GQA question is absent from the official split: {sample_id}")
        question = str(source.get("question", "")).strip()
        fixture_prompt = str(row.get("text", ""))
        expected_fixture_prompt = f"{question}{_LLAVA_SUFFIX}"
        if fixture_prompt != expected_fixture_prompt:
            raise ValueError(f"LLaVA GQA prompt differs from official question {sample_id}")
        image_id = str(source.get("imageId", "")).strip()
        image_name = str(row.get("image", "")).strip()
        if not image_id or image_name != f"{image_id}.jpg":
            raise ValueError(f"LLaVA GQA image binding differs for question {sample_id}")
        answer = str(source.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"official GQA question has no answer: {sample_id}")
        normalized_answer = normalize_gqa_prediction(answer)
        if normalized_answer != answer:
            raise ValueError(f"official GQA answer is not already normalized: {sample_id}")
        examples.append(
            EvaluationExample(
                id=sample_id,
                prompt=render_benchmark_prompt(spec, {"question": question}),
                images=(root / image_name,),
                references=(answer,),
                metadata={
                    "image_id": image_id,
                    "structural_type": source.get("types", {}).get("structural"),
                    "semantic_type": source.get("types", {}).get("semantic"),
                    "detailed_type": source.get("types", {}).get("detailed"),
                },
            )
        )
    if set(questions) != seen:
        raise ValueError("official GQA split and LLaVA fixture contain different question ID sets")
    return examples


def _extract_images(
    archive: Path,
    examples: list[EvaluationExample],
    destination: Path,
    *,
    progress: Callable[[str, int, int], None] | None,
) -> list[dict[str, Any]]:
    names = sorted({example.images[0].name for example in examples})
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive) as bundle:
        by_basename: dict[str, zipfile.ZipInfo] = {}
        wanted = set(names)
        for info in bundle.infolist():
            name = Path(info.filename).name
            if name not in wanted or info.is_dir():
                continue
            if name in by_basename:
                raise ValueError(f"GQA image archive contains duplicate basename: {name}")
            if info.file_size <= 0 or info.file_size > 100 * 1024**2:
                raise ValueError(f"GQA image has an invalid archived size: {info.filename}")
            by_basename[name] = info
        missing = wanted - set(by_basename)
        if missing:
            raise ValueError(f"GQA image archive lacks referenced images: {sorted(missing)[:8]}")
        for index, name in enumerate(names, start=1):
            with bundle.open(by_basename[name]) as stream:
                payload = stream.read()
            target = destination / name
            atomic_write_bytes(target, payload)
            with Image.open(target) as image:
                image.load()
                pixel_sha256 = canonical_rgb_sha256(image.convert("RGB"))
                image_format = str(image.format or "unknown")
            records.append(
                {
                    "file": f"images/{name}",
                    "sha256": sha256_file(target),
                    "pixel_sha256": pixel_sha256,
                    "size_bytes": target.stat().st_size,
                    "format": image_format,
                }
            )
            if progress is not None and (index % 250 == 0 or index == len(names)):
                progress("extract-images", index, len(names))
    return records


def materialize_gqa_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Download official archives and emit the exact LLaVA test-dev split locally."""

    if spec.id != "gqa" or spec.split != "testdev_balanced":
        raise ValueError("GQA materialization requires the testdev_balanced benchmark")
    if spec.annotations.kind != "http" or not spec.annotations.sha256:
        raise ValueError("GQA questions must be a checksummed HTTP archive")
    if spec.images is None or spec.images.kind != "http" or not spec.images.sha256:
        raise ValueError("GQA images must be a checksummed HTTP archive")
    fixture_source = _protocol_source(spec, "llava-v1.5-eval")
    cache_root = Path(cache_dir)
    questions_archive = download_http(
        spec.annotations.location,
        cache_root / "benchmarks" / spec.annotations.id / "questions1.2.zip",
        sha256=spec.annotations.sha256,
    )
    images_archive = download_http(
        spec.images.location,
        cache_root / "benchmarks" / spec.images.id / "images.zip",
        sha256=spec.images.sha256,
    )
    fixture_archive = download_http(
        fixture_source.location,
        cache_root / "protocols" / fixture_source.id / "eval.zip",
        sha256=fixture_source.sha256,
    )
    for source in spec.protocol_sources:
        if source.id == fixture_source.id:
            continue
        if source.kind != "http" or not source.sha256:
            raise ValueError(f"GQA protocol source {source.id} must be checksummed HTTP")
        source_name = Path(urlparse(source.location).path).name or source.id
        download_http(
            source.location,
            cache_root / "protocols" / source.id / source_name,
            sha256=source.sha256,
        )
    with zipfile.ZipFile(questions_archive) as bundle:
        questions = _load_json_member(bundle, _unique_member(bundle, suffix=_QUESTION_SUFFIX))
    with zipfile.ZipFile(fixture_archive) as bundle:
        llava_rows = _load_jsonl_member(bundle, _LLAVA_QUESTION_MEMBER)
    examples = build_gqa_examples(questions, llava_rows, spec=spec)
    if len(examples) != _EXPECTED_ROWS:
        raise ValueError(f"GQA test-dev balanced must contain {_EXPECTED_ROWS} questions")

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        image_records = _extract_images(
            images_archive,
            examples,
            temporary / "images",
            progress=progress,
        )
        examples_path = temporary / "examples.jsonl"
        write_examples(examples, examples_path)
        if progress is not None:
            progress("convert", len(examples), len(examples))
        audit = audit_image_integrity(
            (
                ImageReferenceSet(
                    tuple(temporary / image for image in example.images),
                    spec.id,
                )
                for example in examples
            ),
            image_root=temporary,
            workers=8,
            progress_every=250,
            progress=(
                (lambda complete, total: progress("image-audit", complete, total))
                if progress is not None
                else None
            ),
        )
        audit_payload = {
            "schema_version": 1,
            "benchmark_id": spec.id,
            "protocol_revision": spec.protocol_revision,
            "image_root": "images",
            **audit.to_dict(),
        }
        audit_path = temporary / "image-integrity.json"
        atomic_write_json(audit_path, audit_payload)
        if audit.unique_images != len(image_records) or not audit.passed:
            raise RuntimeError("materialized GQA image integrity audit failed")
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
                "split": spec.split,
                "sample_count": len(examples),
                "unique_image_count": len(image_records),
                "examples_sha256": sha256_file(examples_path),
                "sources": {
                    spec.annotations.id: {
                        "location": spec.annotations.location,
                        "revision": spec.annotations.revision,
                        "sha256": spec.annotations.sha256,
                    },
                    spec.images.id: {
                        "location": spec.images.location,
                        "revision": spec.images.revision,
                        "sha256": spec.images.sha256,
                    },
                    **{
                        source.id: {
                            "location": source.location,
                            "revision": source.revision,
                            "sha256": source.sha256,
                        }
                        for source in spec.protocol_sources
                    },
                },
                "images": image_records,
                "image_integrity": {
                    **audit_payload,
                    "report_sha256": sha256_file(audit_path),
                },
            },
        )
        temporary.chmod(0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
