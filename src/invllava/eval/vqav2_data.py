"""Checksummed, relocatable materialization of VQAv2 validation."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.identifiers import benchmark_protocol_id
from invllava.config.schema import BenchmarkSpec, DataSource
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import prepare_vqav2, write_examples

_EXPECTED_ROWS = 214_354
_EXPECTED_IMAGES = 40_504
_QUESTION_MEMBER = "v2_OpenEnded_mscoco_val2014_questions.json"
_ANSWER_MEMBER = "v2_mscoco_val2014_annotations.json"


def _protocol_source(spec: BenchmarkSpec, source_id: str) -> DataSource:
    matches = [source for source in spec.protocol_sources if source.id == source_id]
    if len(matches) != 1:
        raise ValueError(f"VQAv2 requires one protocol source named {source_id}")
    source = matches[0]
    if source.kind != "http" or not source.sha256:
        raise ValueError(f"VQAv2 protocol source {source_id} must be checksummed HTTP")
    return source


def _member_bytes(archive: Path, member: str, *, maximum_bytes: int) -> bytes:
    with zipfile.ZipFile(archive) as bundle:
        matches = [item for item in bundle.infolist() if Path(item.filename).name == member]
        if len(matches) != 1:
            raise ValueError(f"archive must contain exactly one {member}")
        info = matches[0]
        if info.file_size <= 0 or info.file_size > maximum_bytes:
            raise ValueError(f"archive member has an invalid size: {info.filename}")
        payload = bundle.read(info)
    json.loads(payload)
    return payload


def _extract_images(
    archive: Path,
    names: list[str],
    destination: Path,
    *,
    progress: Callable[[str, int, int], None] | None,
) -> list[dict[str, object]]:
    destination.mkdir(parents=True)
    wanted = set(names)
    records: list[dict[str, object]] = []
    with zipfile.ZipFile(archive) as bundle:
        members: dict[str, zipfile.ZipInfo] = {}
        for info in bundle.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in wanted:
                continue
            if name in members:
                raise ValueError(f"COCO archive contains duplicate image basename: {name}")
            if info.file_size <= 0 or info.file_size > 100 * 1024**2:
                raise ValueError(f"COCO image has an invalid archived size: {info.filename}")
            members[name] = info
        missing = wanted - set(members)
        if missing:
            raise ValueError(f"COCO archive lacks referenced images: {sorted(missing)[:8]}")
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
            if progress is not None and (index % 500 == 0 or index == len(names)):
                progress("extract-images", index, len(names))
    return records


def materialize_vqav2_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    """Acquire official VQAv2/COCO sources and seal the complete validation set."""

    if spec.id != "vqav2-val" or spec.split != "val2014":
        raise ValueError("VQAv2 materialization requires the public validation benchmark")
    if spec.annotations.kind != "http" or not spec.annotations.sha256:
        raise ValueError("VQAv2 questions must be a checksummed HTTP archive")
    if spec.images is None or spec.images.kind != "http" or not spec.images.sha256:
        raise ValueError("VQAv2 images must be a checksummed HTTP archive")
    answers_source = _protocol_source(spec, "vqav2-val-answers")
    evaluator_source = _protocol_source(spec, "vqav2-official-evaluator")
    cache_root = Path(cache_dir) / "benchmarks"
    questions_archive = download_http(
        spec.annotations.location,
        cache_root / spec.annotations.id / "questions.zip",
        sha256=spec.annotations.sha256,
    )
    answers_archive = download_http(
        answers_source.location,
        cache_root / answers_source.id / "annotations.zip",
        sha256=answers_source.sha256,
    )
    images_archive = download_http(
        spec.images.location,
        cache_root / spec.images.id / "val2014.zip",
        sha256=spec.images.sha256,
    )
    evaluator_path = download_http(
        evaluator_source.location,
        cache_root / evaluator_source.id / "vqaEval.py",
        sha256=evaluator_source.sha256,
    )
    questions_payload = _member_bytes(
        questions_archive, _QUESTION_MEMBER, maximum_bytes=128 * 1024**2
    )
    answers_payload = _member_bytes(answers_archive, _ANSWER_MEMBER, maximum_bytes=256 * 1024**2)

    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        sources = temporary / "sources"
        sources.mkdir()
        questions_path = sources / _QUESTION_MEMBER
        answers_path = sources / _ANSWER_MEMBER
        local_evaluator_path = sources / "vqaEval.py"
        atomic_write_bytes(questions_path, questions_payload)
        atomic_write_bytes(answers_path, answers_payload)
        atomic_write_bytes(local_evaluator_path, evaluator_path.read_bytes())
        examples = prepare_vqav2(
            questions_path,
            Path("images"),
            coco_split="val2014",
            spec=spec,
            annotations_path=answers_path,
        )
        if len(examples) != _EXPECTED_ROWS:
            raise ValueError(f"VQAv2 validation must contain {_EXPECTED_ROWS} questions")
        names = sorted({example.images[0].name for example in examples})
        if len(names) != _EXPECTED_IMAGES:
            raise ValueError(f"VQAv2 validation must reference {_EXPECTED_IMAGES} images")
        image_records = _extract_images(
            images_archive,
            names,
            temporary / "images",
            progress=progress,
        )
        examples_path = temporary / "examples.jsonl"
        write_examples(examples, examples_path)
        if progress is not None:
            progress("convert", len(examples), len(examples))
        audit = audit_image_integrity(
            (
                ImageReferenceSet(tuple(temporary / image for image in example.images), spec.id)
                for example in examples
            ),
            image_root=temporary,
            workers=8,
            progress_every=500,
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
        if audit.unique_images != _EXPECTED_IMAGES or not audit.passed:
            raise RuntimeError("materialized VQAv2 image integrity audit failed")
        all_sources = (spec.annotations, spec.images, *spec.protocol_sources)
        atomic_write_json(
            temporary / "manifest.json",
            {
                "format": "invllava-eval-dataset-v1",
                "invllava_version": __version__,
                "execution_source_sha256": optional_sha256_environment(
                    "INVLLAVA_EXECUTION_SOURCE_SHA256"
                ),
                "benchmark_id": spec.id,
                "protocol_id": benchmark_protocol_id(spec),
                "protocol_revision": spec.protocol_revision,
                "protocol_config_sha256": config_sha256,
                "conversation_template": spec.conversation_template,
                "split": spec.split,
                "sample_count": len(examples),
                "unique_image_count": len(image_records),
                "examples_sha256": sha256_file(examples_path),
                "source_files": {
                    _QUESTION_MEMBER: sha256_file(questions_path),
                    _ANSWER_MEMBER: sha256_file(answers_path),
                    "vqaEval.py": sha256_file(local_evaluator_path),
                },
                "sources": {
                    source.id: {
                        "location": source.location,
                        "revision": source.revision,
                        "sha256": source.sha256,
                    }
                    for source in all_sources
                },
                "images": image_records,
                "image_integrity": {
                    **audit_payload,
                    "report_sha256": sha256_file(audit_path),
                },
            },
        )
        os.replace(temporary, target)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
