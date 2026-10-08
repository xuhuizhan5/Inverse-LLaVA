#!/usr/bin/env python3
"""Verify GQA prompts, references, image bindings, and official accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file

_QUESTION_SUFFIX = "testdev_balanced_questions.json"
_LLAVA_QUESTION_MEMBER = "gqa/llava_gqa_testdev_balanced.jsonl"
_LLAVA_ANSWER_MEMBER = "gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-13b.jsonl"
_METRIC = re.compile(r"^(Binary|Open|Accuracy):\s+([0-9.]+)%\s*$", re.MULTILINE)


def _member_ending(bundle: zipfile.ZipFile, suffix: str) -> zipfile.ZipInfo:
    matches = [info for info in bundle.infolist() if info.filename.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain one member ending with {suffix}")
    return matches[0]


def _jsonl_member(bundle: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in bundle.read(member).decode("utf-8").splitlines()]


def _archive_image_index(bundle: zipfile.ZipFile, names: set[str]) -> dict[str, zipfile.ZipInfo]:
    result = {}
    for info in bundle.infolist():
        basename = Path(info.filename).name
        if basename in names and not info.is_dir():
            if basename in result:
                raise ValueError(f"GQA image archive contains duplicate basename: {basename}")
            result[basename] = info
    if set(result) != names:
        raise ValueError(f"GQA image archive lacks {len(names - set(result))} referenced images")
    return result


def _average(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--questions-archive", type=Path, required=True)
    parser.add_argument("--images-archive", type=Path, required=True)
    parser.add_argument("--llava-archive", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    sys.path.insert(0, str(args.source))
    from invllava.eval.datasets import load_examples
    from invllava.eval.protocols.gqa import GQAProtocol, normalize_gqa_prediction
    from invllava.prompting import format_vicuna_v1_user_prompt

    with zipfile.ZipFile(args.questions_archive) as bundle:
        questions_info = _member_ending(bundle, _QUESTION_SUFFIX)
        questions = json.loads(bundle.read(questions_info))
    if not isinstance(questions, dict) or len(questions) != 12_578:
        raise ValueError("official GQA test-dev balanced must contain 12,578 questions")
    with zipfile.ZipFile(args.llava_archive) as bundle:
        prompts = _jsonl_member(bundle, _LLAVA_QUESTION_MEMBER)
        released_answers = _jsonl_member(bundle, _LLAVA_ANSWER_MEMBER)
    if len(prompts) != len(released_answers) or len(prompts) != 12_578:
        raise ValueError("released LLaVA GQA fixtures must contain 12,578 rows")

    examples = load_examples(args.examples)
    example_index = {example.id: example for example in examples}
    prompt_index = {str(row["question_id"]): row for row in prompts}
    answer_index = {str(row["question_id"]): row for row in released_answers}
    id_sets = (set(questions), set(example_index), set(prompt_index), set(answer_index))
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("official, released, and materialized GQA question IDs differ")

    prompt_mismatches: list[str] = []
    reference_mismatches: list[str] = []
    image_mismatches: list[str] = []
    released_prompt_mismatches: list[str] = []
    normalized_predictions: dict[str, str] = {}
    direct_scores: list[int] = []
    direct_by_answer_type: dict[str, list[int]] = defaultdict(list)
    expected_image_names: set[str] = set()
    materialized_by_image: dict[str, Path] = {}
    for sample_id, question in questions.items():
        example = example_index[sample_id]
        prompt_row = prompt_index[sample_id]
        answer_row = answer_index[sample_id]
        expected_text = (
            f"{question['question']}\nAnswer the question using a single word or phrase."
        )
        if str(prompt_row["text"]) != expected_text or example.prompt != (
            format_vicuna_v1_user_prompt(f"<image>\n{expected_text}")
        ):
            prompt_mismatches.append(sample_id)
        reference = str(question["answer"])
        if example.references != (reference,):
            reference_mismatches.append(sample_id)
        expected_image = f"{question['imageId']}.jpg"
        expected_image_names.add(expected_image)
        if (
            str(prompt_row["image"]) != expected_image
            or len(example.images) != 1
            or example.images[0].name != expected_image
            or not example.images[0].is_file()
        ):
            image_mismatches.append(sample_id)
        previous_image = materialized_by_image.setdefault(expected_image, example.images[0])
        if previous_image != example.images[0]:
            image_mismatches.append(sample_id)
        if str(answer_row["prompt"]) != expected_text:
            released_prompt_mismatches.append(sample_id)
        prediction = normalize_gqa_prediction(str(answer_row["text"]))
        normalized_predictions[sample_id] = prediction
        correct = int(prediction == reference)
        direct_scores.append(correct)
        answer_type = "open" if question["types"]["structural"] == "query" else "binary"
        direct_by_answer_type[answer_type].append(correct)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_images = {Path(row["file"]).name: row for row in manifest["images"]}
    source_image_mismatches: list[str] = []
    with zipfile.ZipFile(args.images_archive) as bundle:
        archive_images = _archive_image_index(bundle, expected_image_names)
        for image_name in sorted(expected_image_names):
            source_digest = hashlib.sha256(bundle.read(archive_images[image_name])).hexdigest()
            materialized = materialized_by_image[image_name]
            record = manifest_images.get(image_name)
            if (
                record is None
                or record.get("sha256") != source_digest
                or sha256_file(materialized) != source_digest
            ):
                source_image_mismatches.append(image_name)

    local = GQAProtocol("golden").score(normalized_predictions, examples)
    direct_accuracy = _average(direct_scores)
    direct_answer_types = {
        name: _average(values) for name, values in sorted(direct_by_answer_type.items())
    }
    aggregation_mismatches: list[dict[str, Any]] = []
    if not math.isclose(local.value, direct_accuracy, rel_tol=0.0, abs_tol=1e-15):
        aggregation_mismatches.append(
            {"metric": "accuracy", "local": local.value, "official_direct": direct_accuracy}
        )
    if local.details["answer_type_accuracy"] != direct_answer_types:
        aggregation_mismatches.append(
            {
                "metric": "answer_type_accuracy",
                "local": local.details["answer_type_accuracy"],
                "official_direct": direct_answer_types,
            }
        )

    official_predictions = [
        {"questionId": sample_id, "prediction": normalized_predictions[sample_id]}
        for sample_id in questions
    ]
    with tempfile.TemporaryDirectory(prefix="invllava-gqa-golden-") as directory:
        temporary = Path(directory)
        question_path = temporary / "testdev_balanced_questions.json"
        prediction_path = temporary / "testdev_balanced_predictions.json"
        question_path.write_text(json.dumps(questions), encoding="utf-8")
        prediction_path.write_text(json.dumps(official_predictions), encoding="utf-8")
        process = subprocess.run(
            [
                sys.executable,
                str(args.official_evaluator.resolve()),
                "--tier",
                "testdev_balanced",
                "--questions",
                str(question_path),
                "--predictions",
                str(prediction_path),
            ],
            cwd=temporary,
            check=True,
            capture_output=True,
            text=True,
        )
    official_metrics = {
        name.lower(): float(value) / 100.0 for name, value in _METRIC.findall(process.stdout)
    }
    expected_official = {"accuracy": direct_accuracy, **direct_answer_types}
    for name, expected in expected_official.items():
        observed = official_metrics.get(name)
        if observed is None or round(expected * 100, 2) != round(observed * 100, 2):
            aggregation_mismatches.append(
                {"metric": f"official_cli_{name}", "local": expected, "official": observed}
            )

    count_contract = (
        len(examples) == manifest.get("sample_count") == 12_578
        and len(expected_image_names) == manifest.get("unique_image_count")
        and len(manifest_images) == len(expected_image_names)
        and manifest.get("image_integrity", {}).get("unique_images") == len(expected_image_names)
        and manifest.get("image_integrity", {}).get("passed") is True
    )
    mismatch_count = sum(
        len(values)
        for values in (
            prompt_mismatches,
            reference_mismatches,
            image_mismatches,
            source_image_mismatches,
            released_prompt_mismatches,
            aggregation_mismatches,
        )
    )
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "gqa",
        "verifier_sha256": sha256_file(Path(__file__)),
        "upstream_revision": args.upstream_revision,
        "official_evaluator_sha256": sha256_file(args.official_evaluator),
        "questions_archive_sha256": sha256_file(args.questions_archive),
        "questions_member": questions_info.filename,
        "images_archive_sha256": sha256_file(args.images_archive),
        "llava_archive_sha256": sha256_file(args.llava_archive),
        "examples_sha256": sha256_file(args.examples),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(examples),
        "unique_image_count": len(expected_image_names),
        "released_llava_1_5_13b_accuracy": local.value,
        "released_llava_1_5_13b_answer_type_accuracy": direct_answer_types,
        "official_cli_metrics": official_metrics,
        "mismatch_counts": {
            "prompts": len(prompt_mismatches),
            "references": len(reference_mismatches),
            "image_bindings": len(image_mismatches),
            "source_images": len(source_image_mismatches),
            "released_prompts": len(released_prompt_mismatches),
            "aggregation": len(aggregation_mismatches),
        },
        "mismatch_examples": {
            "prompts": prompt_mismatches[:20],
            "references": reference_mismatches[:20],
            "image_bindings": image_mismatches[:20],
            "source_images": source_image_mismatches[:20],
            "released_prompts": released_prompt_mismatches[:20],
            "aggregation": aggregation_mismatches[:20],
        },
        "count_contract_passed": count_contract,
        "passed": mismatch_count == 0 and count_contract,
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"GQA golden failed with {mismatch_count} mismatches")
    print(
        f"GQA golden passed: {len(examples)} samples, {len(expected_image_names)} images, "
        f"released accuracy {local.value:.5f}"
    )


if __name__ == "__main__":
    main()
