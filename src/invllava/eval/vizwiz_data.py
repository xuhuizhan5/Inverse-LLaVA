"""Checksummed, relocatable materialization of the public VizWiz VQA test release."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import prepare_vizwiz, write_examples
from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt

_EXPECTED_ROWS = 8_000


def load_llava_questions(archive: Path) -> dict[str, str]:
    """Read the bounded question fixture, rejecting duplicate bindings."""
    member = "vizwiz/llava_test.jsonl"
    with zipfile.ZipFile(archive) as bundle:
        info = bundle.getinfo(member)
        if info.file_size > 32 * 1024**2:
            raise ValueError("VizWiz prompt fixture exceeds the 32 MiB bound")
        payload = bundle.read(info)
    rows = [json.loads(line) for line in payload.splitlines()]
    by_image = {str(row["image"]): row for row in rows}
    if len(by_image) != len(rows) or len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate image or question ID in the LLaVA VizWiz fixture")
    return {image: str(row["text"]) for image, row in by_image.items()}


def verify_llava_questions(examples: list[EvaluationExample], archive: Path) -> dict[str, object]:
    """Check rendered prompts against the independently released LLaVA fixture."""
    questions = load_llava_questions(archive)
    if len({example.id for example in examples}) != len(examples):
        raise ValueError("duplicate VizWiz example ID")
    if set(questions) != {example.id for example in examples}:
        raise ValueError("VizWiz IDs disagree with the released LLaVA fixture")
    for example in examples:
        expected = format_vicuna_v1_user_prompt("<image>\n" + questions[example.id])
        if example.prompt != expected or example.images[0].name != example.id:
            raise ValueError(f"VizWiz prompt/image differs from the LLaVA fixture: {example.id}")
    member = "vizwiz/llava_test.jsonl"
    with zipfile.ZipFile(archive) as bundle:
        fixture_sha256 = hashlib.sha256(bundle.read(member)).hexdigest()
    return {
        "archive_sha256": sha256_file(archive),
        "member": member,
        "fixture_sha256": fixture_sha256,
        "verified_prompts": len(examples),
        "prompt_mismatches": 0,
    }


def _extract_images(
    archive: Path,
    examples: list[EvaluationExample],
    destination: Path,
    *,
    progress: Callable[[str, int, int], None] | None,
) -> list[dict[str, object]]:
    names = sorted({example.images[0].name for example in examples})
    if len(names) != len(examples):
        raise ValueError("VizWiz test release must bind one unique image to every question")
    destination.mkdir(parents=True)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as bundle:
        wanted = set(names)
        members: dict[str, zipfile.ZipInfo] = {}
        for info in bundle.infolist():
            name = Path(info.filename).name
            if name not in wanted or info.is_dir():
                continue
            if name in members:
                raise ValueError(f"VizWiz archive contains duplicate image basename: {name}")
            if info.file_size <= 0 or info.file_size > 100 * 1024**2:
                raise ValueError(f"VizWiz image has an invalid archived size: {info.filename}")
            members[name] = info
        missing = wanted - set(members)
        if missing:
            raise ValueError(f"VizWiz archive lacks referenced images: {sorted(missing)[:8]}")

        for index, name in enumerate(names, start=1):
            payload = bundle.read(members[name])
            target = destination / name
            atomic_write_bytes(target, payload)
            with Image.open(target) as image:
                image.load()
                image_format = str(image.format or "unknown")
                pixel_sha256 = canonical_rgb_sha256(image.convert("RGB"))
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


def materialize_vizwiz_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Download the released answers/images and emit a fully audited local dataset."""

    if spec.id != "vizwiz" or spec.split != "test":
        raise ValueError("VizWiz materialization requires the public test benchmark")
    if spec.annotations.kind != "http" or not spec.annotations.sha256:
        raise ValueError("VizWiz answers must be a checksummed HTTP source")
    if spec.images is None or spec.images.kind != "http" or not spec.images.sha256:
        raise ValueError("VizWiz images must be a checksummed HTTP archive")

    cache_root = Path(cache_dir)
    annotations = download_http(
        spec.annotations.location,
        cache_root / "benchmarks" / spec.annotations.id / "VQA_test.json",
        sha256=spec.annotations.sha256,
    )
    sources = [source for source in spec.protocol_sources if source.id == "llava-v1.5-eval"]
    if len(sources) != 1 or sources[0].kind != "http" or not sources[0].sha256:
        raise ValueError("VizWiz requires one checksummed LLaVA evaluation archive")
    fixture_source = sources[0]
    fixture_archive = download_http(
        fixture_source.location,
        cache_root / "benchmarks" / fixture_source.id / "eval.zip",
        sha256=fixture_source.sha256,
    )
    examples = prepare_vizwiz(
        annotations,
        Path("images"),
        spec=spec,
        llava_questions=load_llava_questions(fixture_archive),
    )
    if len(examples) != _EXPECTED_ROWS:
        raise ValueError(f"VizWiz public test release must contain {_EXPECTED_ROWS} questions")
    prompt_fixture = verify_llava_questions(examples, fixture_archive)
    images_archive = download_http(
        spec.images.location,
        cache_root / "benchmarks" / spec.images.id / "test.zip",
        sha256=spec.images.sha256,
    )

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
            raise RuntimeError("materialized VizWiz image integrity audit failed")
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
                    fixture_source.id: {
                        "location": fixture_source.location,
                        "revision": fixture_source.revision,
                        "sha256": fixture_source.sha256,
                    },
                },
                "prompt_fixture": prompt_fixture,
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
