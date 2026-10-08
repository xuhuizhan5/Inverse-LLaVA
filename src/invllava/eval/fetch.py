"""Pinned Hugging Face benchmark materialization into the shared local schema."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from invllava import __version__
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment, sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.data.audit import ImageReferenceSet, audit_image_integrity, canonical_rgb_sha256
from invllava.data.download import download_http
from invllava.eval.datasets import write_examples
from invllava.eval.prompts import format_choices, render_benchmark_prompt
from invllava.eval.types import EvaluationExample

_SCIENCEQA_QUESTION_MEMBER = "scienceqa/llava_test_CQM-A.json"


def _identifier(doc: dict[str, Any], index: int) -> str:
    for key in ("id", "index", "question_id", "pid"):
        if key in doc and doc[key] is not None:
            return str(doc[key])
    return str(index)


def _save_image(
    image: Any,
    *,
    sample_id: str,
    physical_root: Path,
    logical_root: Path,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(image, Image.Image):
        raise TypeError(f"decoded benchmark image is not PIL for sample {sample_id}")
    name = hashlib.sha256(sample_id.encode()).hexdigest() + ".png"
    physical = physical_root / name
    rgb = image.convert("RGB")
    pixel_sha256 = canonical_rgb_sha256(rgb)
    if physical.exists():
        with Image.open(physical) as existing:
            if canonical_rgb_sha256(existing.convert("RGB")) != pixel_sha256:
                raise ValueError(f"image key {sample_id} resolves to inconsistent pixels")
    else:
        rgb.save(physical, format="PNG", compress_level=1)
    return logical_root / name, {
        "file": f"images/{name}",
        "sha256": sha256_file(physical),
        "pixel_sha256": pixel_sha256,
        "size_bytes": physical.stat().st_size,
    }


def _choices(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("multiple-choice benchmark row has no choices")
    return tuple(str(item) for item in value)


def _scienceqa_expected_source_prompt(doc: dict[str, Any]) -> str:
    """Reconstruct LLaVA's released converter output for fixture auditing."""

    context = str(doc.get("hint") or "").strip()
    question = str(doc["question"]).strip()
    body = f"Context: {context}\n{question}" if context else question
    body = f"{body}\n{format_choices(_choices(doc['choices']))}"
    body = body.replace("  ", " ").strip()
    return f"<image>\n{body}" if doc.get("image") is not None else body


def _validated_scienceqa_record(
    doc: dict[str, Any], official: dict[str, Any] | None
) -> tuple[str, str, str]:
    if official is None:
        raise ValueError("ScienceQA conversion requires the official LLaVA question record")
    conversations = official.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("official ScienceQA record has an invalid conversation")
    human = conversations[0]
    assistant = conversations[1]
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError("official ScienceQA record has unexpected conversation roles")
    source_prompt = str(human.get("value", ""))
    expected_prompt = _scienceqa_expected_source_prompt(doc)
    if source_prompt != expected_prompt:
        raise ValueError(f"ScienceQA mirror prompt differs for source row {official.get('id')}")
    expected_answer = chr(ord("A") + int(doc["answer"]))
    if assistant.get("value") != expected_answer:
        raise ValueError(f"ScienceQA mirror answer differs for source row {official.get('id')}")
    has_image = doc.get("image") is not None
    if ("image" in official) != has_image:
        raise ValueError(f"ScienceQA mirror image flag differs for source row {official.get('id')}")
    sample_id = str(official.get("id", "")).strip()
    if not sample_id:
        raise ValueError("official ScienceQA record has no id")
    return sample_id, source_prompt, expected_answer


def _load_scienceqa_questions(spec: BenchmarkSpec, cache_dir: Path) -> list[dict[str, Any]]:
    source = spec.annotations
    if source.kind != "http" or not source.sha256:
        raise ValueError("ScienceQA requires the checksummed official LLaVA evaluation archive")
    archive = download_http(
        source.location,
        cache_dir / "protocols" / source.id / "eval.zip",
        sha256=source.sha256,
    )
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(_SCIENCEQA_QUESTION_MEMBER)
        except KeyError as error:
            raise ValueError(
                "official LLaVA archive lacks the ScienceQA question fixture"
            ) from error
        if info.file_size > 64 * 1024**2:
            raise ValueError("ScienceQA question fixture exceeds the 64 MiB safety bound")
        with bundle.open(info) as stream:
            value = json.load(stream)
    if not isinstance(value, list) or not value:
        raise ValueError("official ScienceQA question fixture is empty or malformed")
    return value


def _load_textvqa_annotations(spec: BenchmarkSpec, cache_dir: Path) -> dict[str, dict[str, Any]]:
    source = spec.annotations
    if source.kind != "http" or not source.sha256:
        raise ValueError("TextVQA requires the checksummed official v0.5.1 annotations")
    annotation_path = download_http(
        source.location,
        cache_dir / "protocols" / source.id / "TextVQA_0.5.1_val.json",
        sha256=source.sha256,
    )
    value = json.loads(annotation_path.read_text(encoding="utf-8"))
    rows = value.get("data") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 5_000:
        raise ValueError("official TextVQA validation annotations must contain 5,000 rows")
    records = {str(row["question_id"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError("official TextVQA annotations contain duplicate question IDs")
    return records


def _convert(
    spec: BenchmarkSpec,
    doc: dict[str, Any],
    index: int,
    *,
    physical_image_root: Path,
    logical_image_root: Path,
    official_record: dict[str, Any] | None = None,
) -> tuple[EvaluationExample | None, dict[str, Any] | None]:
    benchmark_id = spec.id
    if benchmark_id == "scienceqa-img":
        sample_id, source_prompt, answer = _validated_scienceqa_record(doc, official_record)
        if doc.get("image") is None:
            return None, None
    else:
        sample_id = _identifier(doc, index)
    image_key = str(doc["image_id"]) if benchmark_id == "textvqa" else sample_id
    image_path, image_record = _save_image(
        doc.get("image"),
        sample_id=image_key,
        physical_root=physical_image_root,
        logical_root=logical_image_root,
    )
    if benchmark_id == "ai2d":
        choices = _choices(doc["options"])
        answer = chr(ord("A") + int(doc["answer"]))
        example = EvaluationExample(
            id=sample_id,
            prompt=render_benchmark_prompt(
                spec,
                {"question": str(doc["question"]), "choices": format_choices(choices)},
            ),
            images=(image_path,),
            references=(answer,),
            choices=choices,
        )
    elif benchmark_id == "scienceqa-img":
        choices = _choices(doc["choices"])
        if not source_prompt.startswith("<image>\n"):
            raise ValueError(f"official image prompt lacks <image> for source row {sample_id}")
        example = EvaluationExample(
            id=sample_id,
            prompt=render_benchmark_prompt(
                spec,
                {"question": source_prompt.removeprefix("<image>\n")},
            ),
            images=(image_path,),
            references=(answer,),
            choices=choices,
        )
    elif benchmark_id == "mmstar":
        # The official dataset's question already contains its options. Keep it
        # byte-for-byte and provide label placeholders only to bound extraction.
        choices = tuple(str(doc[key]) for key in "ABCD" if key in doc)
        if not choices:
            choices = ("A", "B", "C", "D")
        example = EvaluationExample(
            id=sample_id,
            prompt=render_benchmark_prompt(
                spec,
                {"question": str(doc["question"]).strip()},
            ),
            images=(image_path,),
            references=(str(doc["answer"]).strip().upper(),),
            choices=choices,
            metadata={
                "category": doc.get("category"),
                "l2_category": doc.get("l2_category"),
            },
        )
    elif benchmark_id == "ocrbench":
        raw_answer = doc["answer"]
        answers = raw_answer if isinstance(raw_answer, (list, tuple)) else (raw_answer,)
        example = EvaluationExample(
            id=sample_id,
            prompt=render_benchmark_prompt(
                spec,
                {"question": str(doc["question"]).strip()},
            ),
            images=(image_path,),
            references=tuple(str(answer) for answer in answers),
            metadata={
                "dataset": str(doc["dataset"]),
                "question_type": str(doc["question_type"]),
            },
        )
    elif benchmark_id == "textvqa":
        if official_record is None:
            raise ValueError("TextVQA conversion requires the official annotation row")
        comparisons = {
            "question_id": (str(doc["question_id"]), str(official_record["question_id"])),
            "image_id": (str(doc["image_id"]), str(official_record["image_id"])),
            "question": (str(doc["question"]), str(official_record["question"])),
            "answers": (
                tuple(str(answer) for answer in doc["answers"]),
                tuple(str(answer) for answer in official_record["answers"]),
            ),
        }
        mismatches = [name for name, values in comparisons.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                f"TextVQA mirror differs from official row {sample_id}: {', '.join(mismatches)}"
            )
        ocr_tokens = tuple(str(token) for token in doc.get("ocr_tokens", ()))
        example = EvaluationExample(
            id=sample_id,
            prompt=render_benchmark_prompt(
                spec,
                {
                    "question": str(doc["question"]).capitalize(),
                    "ocr_tokens": ", ".join(ocr_tokens),
                },
            ),
            images=(image_path,),
            references=tuple(str(answer) for answer in official_record["answers"]),
            metadata={
                "image_id": str(doc["image_id"]),
                "ocr_token_count": len(ocr_tokens),
            },
        )
    else:
        raise ValueError(f"no pinned Hugging Face converter for {benchmark_id}")
    return example, image_record


def materialize_huggingface_benchmark(
    spec: BenchmarkSpec,
    destination: str | Path,
    *,
    cache_dir: str | Path,
    config_sha256: str,
    progress: Callable[[str, int, int], None] | None = None,
) -> Path:
    if spec.id not in {"ai2d", "scienceqa-img", "mmstar", "ocrbench", "textvqa"}:
        raise ValueError(f"benchmark {spec.id} is not supported by the pinned HF converter")
    annotation_source = spec.annotations
    dataset_source = spec.images if spec.id in {"scienceqa-img", "textvqa"} else annotation_source
    if dataset_source is None:
        raise ValueError(f"benchmark {spec.id} has no dataset source")
    if dataset_source.kind != "huggingface" or not dataset_source.revision:
        raise ValueError("HF benchmark conversion requires a frozen Hugging Face source")
    if dataset_source.revision == "pending-freeze":
        raise ValueError("freeze the benchmark dataset revision before materialization")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    physical_images = temporary / "images"
    physical_images.mkdir()
    logical_images = Path("images")
    try:
        from datasets import load_dataset

        official_questions = (
            _load_scienceqa_questions(spec, Path(cache_dir)) if spec.id == "scienceqa-img" else None
        )
        official_textvqa = (
            _load_textvqa_annotations(spec, Path(cache_dir)) if spec.id == "textvqa" else None
        )
        if spec.id == "textvqa":
            # The repository also holds roughly 8 GB of train/test shards.
            # The pinned hf:// Parquet boundary retrieves only the requested
            # files and validates that split independently of the repository's
            # train/test metadata.
            dataset = load_dataset(
                "parquet",
                data_files={
                    spec.split: (
                        f"hf://datasets/{dataset_source.location}"
                        f"@{dataset_source.revision}/data/{spec.split}-*.parquet"
                    )
                },
                split=spec.split,
                cache_dir=str(cache_dir),
            )
        else:
            dataset = load_dataset(
                dataset_source.location,
                dataset_source.subset,
                split=spec.split,
                revision=dataset_source.revision,
                cache_dir=str(cache_dir),
            )
        if official_questions is not None and len(official_questions) != len(dataset):
            raise ValueError(
                "official ScienceQA fixture and pinned image mirror have different sizes"
            )
        examples: list[EvaluationExample] = []
        image_records: dict[str, dict[str, Any]] = {}
        for index, row in enumerate(dataset):
            if not isinstance(row, dict):
                row = dict(row)
            example, image_record = _convert(
                spec,
                row,
                index,
                physical_image_root=physical_images,
                logical_image_root=logical_images,
                official_record=(
                    official_questions[index]
                    if official_questions is not None
                    else official_textvqa.get(str(row["question_id"]))
                    if official_textvqa is not None
                    else None
                ),
            )
            if example is not None and image_record is not None:
                examples.append(example)
                previous = image_records.setdefault(str(image_record["file"]), image_record)
                if previous != image_record:
                    raise ValueError(f"inconsistent repeated image record: {image_record['file']}")
            if progress is not None and ((index + 1) % 250 == 0 or index + 1 == len(dataset)):
                progress("convert", index + 1, len(dataset))
        if official_textvqa is not None and len(examples) != len(official_textvqa):
            raise ValueError(
                "pinned TextVQA mirror and official annotations have different sample counts"
            )
        examples_path = temporary / "examples.jsonl"
        write_examples(examples, examples_path)
        image_audit = audit_image_integrity(
            (
                ImageReferenceSet(
                    (physical_images / example.images[0].name,),
                    spec.id,
                )
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
        if image_audit.unique_images == 0 or not image_audit.passed:
            raise RuntimeError("materialized benchmark image integrity audit failed")
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
                "dataset": dataset_source.location,
                "dataset_subset": dataset_source.subset,
                "dataset_revision": dataset_source.revision,
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                "annotation_source": annotation_source.location,
                "annotation_sha256": annotation_source.sha256,
                "split": spec.split,
                "sample_count": len(examples),
                "examples_sha256": sha256_file(examples_path),
                "images": [image_records[key] for key in sorted(image_records)],
                "image_integrity": {
                    **image_audit_payload,
                    "report_sha256": sha256_file(image_audit_path),
                },
            },
        )
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
