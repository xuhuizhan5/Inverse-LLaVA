"""Deterministic conversion from official files to the shared evaluation schema."""

from __future__ import annotations

import base64
import csv
import json
from dataclasses import asdict
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_bytes, atomic_write_text
from invllava.config.schema import BenchmarkSpec
from invllava.eval.prompts import format_choices, render_benchmark_prompt
from invllava.eval.types import EvaluationExample


def write_examples(examples: list[EvaluationExample], destination: str | Path) -> None:
    ids = [example.id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation examples contain duplicate IDs")
    lines = []
    for example in examples:
        value = asdict(example)
        value["images"] = [str(path) for path in example.images]
        lines.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    atomic_write_text(destination, "\n".join(lines) + ("\n" if lines else ""))


def load_examples(path: str | Path) -> list[EvaluationExample]:
    result = []
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            images = []
            for item in value.get("images", []):
                image = Path(item)
                images.append(image if image.is_absolute() else (source.parent / image).resolve())
            result.append(
                EvaluationExample(
                    id=str(value["id"]),
                    prompt=str(value["prompt"]),
                    images=tuple(images),
                    references=tuple(str(item) for item in value.get("references", [])),
                    choices=tuple(str(item) for item in value.get("choices", [])),
                    group_id=value.get("group_id"),
                    metadata=dict(value.get("metadata", {})),
                )
            )
    return result


def prepare_vqav2(
    questions_path: str | Path,
    image_root: str | Path,
    *,
    coco_split: str,
    spec: BenchmarkSpec,
    annotations_path: str | Path | None = None,
) -> list[EvaluationExample]:
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))["questions"]
    question_ids = [int(item["question_id"]) for item in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("VQAv2 questions contain duplicate question IDs")
    annotations_by_id: dict[int, dict[str, object]] = {}
    if annotations_path is not None:
        annotations = json.loads(Path(annotations_path).read_text(encoding="utf-8"))["annotations"]
        annotation_ids = [int(item["question_id"]) for item in annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("VQAv2 annotations contain duplicate question IDs")
        if set(annotation_ids) != set(question_ids):
            raise ValueError("VQAv2 question and annotation inventories differ")
        annotations_by_id = dict(zip(annotation_ids, annotations, strict=True))
    root = Path(image_root)
    examples: list[EvaluationExample] = []
    for item in questions:
        question_id = int(item["question_id"])
        annotation = annotations_by_id.get(question_id)
        if annotation is not None and int(annotation["image_id"]) != int(item["image_id"]):
            raise ValueError(f"VQAv2 question/annotation image mismatch: {question_id}")
        references = (
            tuple(str(answer["answer"]) for answer in annotation["answers"])
            if annotation is not None
            else ()
        )
        if annotation is not None and len(references) != 10:
            raise ValueError(f"VQAv2 annotation {question_id} must contain ten answers")
        metadata = {"image_id": int(item["image_id"]), "coco_split": coco_split}
        if annotation is not None:
            metadata.update(
                {
                    "answer_type": str(annotation["answer_type"]),
                    "question_type": str(annotation["question_type"]),
                }
            )
        examples.append(
            EvaluationExample(
                id=str(question_id),
                prompt=render_benchmark_prompt(spec, {"question": str(item["question"])}),
                images=(root / f"COCO_{coco_split}_{int(item['image_id']):012d}.jpg",),
                references=references,
                metadata=metadata,
            )
        )
    return examples


def prepare_gqa(
    annotation_path: str | Path,
    image_root: str | Path,
    *,
    spec: BenchmarkSpec,
) -> list[EvaluationExample]:
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    root = Path(image_root)
    return [
        EvaluationExample(
            id=str(question_id),
            prompt=render_benchmark_prompt(spec, {"question": str(item["question"])}),
            images=(root / f"{item['imageId']}.jpg",),
            references=(str(item["answer"]),) if "answer" in item else (),
            metadata={"image_id": item["imageId"]},
        )
        for question_id, item in data.items()
    ]


def prepare_mmvet(
    annotation_path: str | Path,
    image_root: str | Path,
    *,
    spec: BenchmarkSpec,
) -> list[EvaluationExample]:
    """Preserve MM-Vet's public IDs and compositional reference syntax."""

    from invllava.artifacts.hashing import verify_sha256

    source = Path(annotation_path)
    if not spec.annotations.sha256:
        raise ValueError("MM-Vet requires a frozen annotation digest")
    verify_sha256(source, spec.annotations.sha256)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("MM-Vet annotations must be a nonempty ID-keyed object")
    root = Path(image_root).resolve()
    examples = []
    for sample_id, row in data.items():
        image = Path(row["imagename"])
        if image.is_absolute() or len(image.parts) != 1 or image.name in {"", ".", ".."}:
            raise ValueError(f"unsafe MM-Vet image name: {image}")
        if not isinstance(row["question"], str) or not isinstance(row["answer"], str):
            raise ValueError(f"MM-Vet question/reference must be text: {sample_id}")
        examples.append(
            EvaluationExample(
                id=sample_id,
                prompt=render_benchmark_prompt(spec, {"question": row["question"]}),
                images=(root / image,),
                references=(row["answer"],),
                group_id=image.name,
                metadata={"capabilities": list(row["capability"])},
            )
        )
    return examples


def prepare_vizwiz(
    annotation_path: str | Path,
    image_root: str | Path,
    *,
    spec: BenchmarkSpec,
    llava_questions: dict[str, str],
) -> list[EvaluationExample]:
    """Join public answers to the exact questions in LLaVA's released fixture."""

    raw = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("annotations"))
    if not isinstance(raw, list):
        raise ValueError("VizWiz annotations must be a list or contain data/annotations")
    root = Path(image_root)
    prefix, marker, suffix = spec.prompt_template.partition("{question}")
    if prefix != "<image>\n" or not marker or not suffix:
        raise ValueError("VizWiz requires the reviewed LLaVA question template")
    examples: list[EvaluationExample] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "image" not in item or "question" not in item:
            raise ValueError(f"VizWiz row {index} is malformed")
        answers = item.get("answers")
        if not isinstance(answers, list) or len(answers) != 10:
            raise ValueError(
                "VizWiz local scoring requires the April 2026 answer-bearing VQA_test.json "
                f"(row {index} has {0 if not isinstance(answers, list) else len(answers)} answers)"
            )
        references = tuple(
            str(answer["answer"] if isinstance(answer, dict) else answer) for answer in answers
        )
        image_name = str(item["image"])
        fixture_text = llava_questions.get(image_name)
        if fixture_text is None or not fixture_text.endswith(suffix):
            raise ValueError(f"VizWiz image lacks a matching LLaVA question: {image_name}")
        question = fixture_text[: -len(suffix)]
        examples.append(
            EvaluationExample(
                id=image_name,
                prompt=render_benchmark_prompt(
                    spec,
                    {"question": question},
                ),
                images=(root / image_name,),
                references=references,
                metadata={
                    "answerable": item.get("answerable"),
                    "answer_type": item.get("answer_type"),
                    "release": "april-2026-public-test-answers",
                },
            )
        )
    if {example.id for example in examples} != set(llava_questions):
        raise ValueError("VizWiz public answer IDs differ from the LLaVA questions")
    return examples


def prepare_mmbench(
    tsv_path: str | Path,
    image_destination: str | Path,
    *,
    spec: BenchmarkSpec,
    logical_image_root: str | Path | None = None,
) -> list[EvaluationExample]:
    image_root = Path(image_destination)
    image_root.mkdir(parents=True, exist_ok=True)
    logical_root = Path(logical_image_root) if logical_image_root is not None else image_root
    examples: list[EvaluationExample] = []
    rows: list[dict[str, str]]
    with Path(tsv_path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))

    encoded_images: dict[str, str] = {}
    for row in rows:
        numeric_id = int(row["index"])
        group_id = str(numeric_id % 1_000_000)
        image_value = str(row["image"])
        if image_value.isdigit():
            if image_value != group_id:
                raise ValueError(
                    f"MMBench row {numeric_id} references image {image_value}, expected {group_id}"
                )
            continue
        previous = encoded_images.setdefault(group_id, image_value)
        if previous != image_value:
            raise ValueError(f"MMBench circular group {group_id} has inconsistent images")

    for row in rows:
        numeric_id = int(row["index"])
        sample_id = str(numeric_id)
        raw_group_id = row.get("g_index")
        group_id = str(
            int(raw_group_id) if raw_group_id not in (None, "", "nan") else numeric_id % 1_000_000
        )
        if group_id not in encoded_images:
            raise ValueError(f"MMBench circular group {group_id} lacks an encoded image")
        physical_image_path = image_root / f"{group_id}.jpg"
        image_path = logical_root / physical_image_path.name
        if not physical_image_path.exists():
            image_payload = base64.b64decode(encoded_images[group_id], validate=True)
            atomic_write_bytes(physical_image_path, image_payload)
        choices = tuple(
            str(row[label])
            for label in ("A", "B", "C", "D")
            if row.get(label) not in (None, "", "nan")
        )
        hint = row.get("hint")
        hint_text = f"Hint: {hint}\n" if hint not in (None, "", "nan") else ""
        examples.append(
            EvaluationExample(
                id=sample_id,
                prompt=render_benchmark_prompt(
                    spec,
                    {
                        "hint_block": hint_text,
                        "raw_hint_block": f"{hint}\n" if hint not in (None, "", "nan") else "",
                        "question": str(row["question"]),
                        "choices": format_choices(choices),
                    },
                ),
                images=(image_path,),
                references=(str(row["answer"]).upper(),) if row.get("answer") else (),
                choices=choices,
                group_id=group_id,
                metadata={
                    "category": row.get("category"),
                    "l2_category": row.get("l2-category"),
                    "split": row.get("split"),
                    "source_file": Path(tsv_path).name,
                },
            )
        )
    return examples


MME_PERCEPTION_CATEGORIES = (
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
)
MME_COGNITION_CATEGORIES = (
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
)

MME_RELEASE_SUFFIX = "Please answer yes or no."
MME_LLAVA_SUFFIX = "Answer the question using a single word or phrase."


def canonicalize_mme_question(question: str) -> str:
    """Reproduce the question text in LLaVA's released MME fixture."""

    value = question.strip()
    if not value.endswith(MME_RELEASE_SUFFIX):
        raise ValueError("MME question lacks the released yes/no suffix")
    value = value[: -len(MME_RELEASE_SUFFIX)].rstrip()
    if not value:
        raise ValueError("MME question is empty after removing its released suffix")
    return f"{value}\n{MME_LLAVA_SUFFIX}"


def _mme_category_root(root: Path) -> Path:
    nested = root / "MME_Benchmark"
    known = set(MME_PERCEPTION_CATEGORIES + MME_COGNITION_CATEGORIES)
    direct_categories = {path.name for path in root.iterdir() if path.is_dir()}
    if direct_categories & known:
        return root
    if nested.is_dir():
        return nested
    raise ValueError(
        f"{root} is not an MME release root: expected category directories directly "
        "or below MME_Benchmark/"
    )


def prepare_mme(
    root: str | Path,
    *,
    domain: str = "all",
    spec: BenchmarkSpec,
) -> list[EvaluationExample]:
    source_root = Path(root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"MME release root does not exist: {source_root}")
    category_root = _mme_category_root(source_root)
    if domain == "perception":
        categories = MME_PERCEPTION_CATEGORIES
    elif domain == "cognition":
        categories = MME_COGNITION_CATEGORIES
    elif domain == "all":
        categories = MME_PERCEPTION_CATEGORIES + MME_COGNITION_CATEGORIES
    else:
        raise ValueError("MME domain must be one of: perception, cognition, all")

    examples: list[EvaluationExample] = []
    for category in categories:
        directory = category_root / category
        if not directory.is_dir():
            raise FileNotFoundError(f"MME category is missing: {directory}")
        separated_images = directory / "images"
        if separated_images.is_dir():
            image_root = separated_images
            annotation_root = directory / "questions_answers_YN"
            if not annotation_root.is_dir():
                raise FileNotFoundError(
                    f"MME category {category} has images/ but no questions_answers_YN/"
                )
        else:
            image_root = directory
            annotation_root = directory

        images = sorted(
            path
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not images:
            raise ValueError(f"MME category {category} contains no supported images")
        for image in images:
            question_file = annotation_root / f"{image.stem}.txt"
            if not question_file.is_file():
                raise FileNotFoundError(f"MME annotation is missing for {image}: {question_file}")
            lines = question_file.read_text(encoding="utf-8").splitlines()
            if len(lines) != 2:
                raise ValueError(
                    f"MME image {image} must have exactly two paired questions, found {len(lines)}"
                )
            for index, line in enumerate(lines):
                fields = line.split("\t")
                if len(fields) != 2:
                    raise ValueError(
                        f"MME annotation {question_file}:{index + 1} must be question<TAB>answer"
                    )
                question, answer = fields
                if not question.strip():
                    raise ValueError(f"MME annotation has an empty question: {question_file}")
                if answer not in {"Yes", "No"}:
                    raise ValueError(
                        f"MME answer must be exactly Yes or No in {question_file}:{index + 1}"
                    )
                group_id = f"{category}/{image.name}"
                domain_name = "perception" if category in MME_PERCEPTION_CATEGORIES else "cognition"
                examples.append(
                    EvaluationExample(
                        id=f"{group_id}/{index}",
                        prompt=render_benchmark_prompt(
                            spec,
                            {"question": canonicalize_mme_question(question)},
                        ),
                        images=(image,),
                        references=(answer,),
                        group_id=group_id,
                        metadata={"category": category, "domain": domain_name},
                    )
                )
    return examples


def validate_image_paths(examples: list[EvaluationExample]) -> None:
    """Recheck every image immediately before a model-backed operation.

    Dataset preparation records a durable full audit. This second pass protects
    against files that were truncated, replaced, or lost later on shared/cloud
    storage, while model weights are still unloaded.
    """

    from invllava.data.audit import ImageReferenceSet, audit_image_integrity

    audit = audit_image_integrity(
        (ImageReferenceSet(example.images, "evaluation-preflight") for example in examples),
        workers=8,
        maximum_failure_details=1,
    )
    if not audit.passed:
        first = audit.failure_details[0] if audit.failure_details else None
        raise RuntimeError(
            "evaluation image preflight failed: "
            f"missing={audit.missing_images}, corrupt={audit.corrupt_images}, "
            f"zero_sized={audit.zero_sized_images}, first={first}"
        )
