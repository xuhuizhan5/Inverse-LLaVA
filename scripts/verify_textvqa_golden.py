#!/usr/bin/env python3
"""Verify paper-era TextVQA prompts, bindings, normalization, and scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file

_PROMPT_MEMBER = "textvqa/llava_textvqa_val_v051_ocr.jsonl"
_ANSWER_MEMBER = "textvqa/answers/llava-v1.5-13b.jsonl"


def _load_reference(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pinned_vqa_eval_metric", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned TextVQA evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(processor: Any, prediction: str, answers: tuple[str, ...]) -> float:
    normalized_prediction = processor(prediction)
    normalized_answers = tuple(processor(answer) for answer in answers)
    values = []
    for index in range(len(normalized_answers)):
        matches = sum(
            normalized_prediction == answer
            for other_index, answer in enumerate(normalized_answers)
            if other_index != index
        )
        values.append(min(1.0, matches / 3.0))
    return sum(values) / len(values)


def _load_jsonl_member(bundle: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in bundle.read(member).decode("utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--upstream-evaluator", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--llava-archive", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    sys.path.insert(0, str(args.source))
    from invllava.eval.datasets import load_examples
    from invllava.eval.protocols.evalai import normalize_textvqa_answer
    from invllava.prompting import VICUNA_V1_SYSTEM

    reference = _load_reference(args.upstream_evaluator)
    reference_processor = reference.EvalAIAnswerProcessor()
    annotation_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations = annotation_payload["data"]
    annotations_by_question = {str(row["question_id"]): row for row in annotations}
    annotations_by_image_question = {
        (str(row["image_id"]), str(row["question"]).lower()): row for row in annotations
    }
    if len(annotations_by_question) != len(annotations) or len(annotations) != 5_000:
        raise ValueError("official TextVQA annotations have an invalid question-ID inventory")
    with zipfile.ZipFile(args.llava_archive) as bundle:
        prompts = _load_jsonl_member(bundle, _PROMPT_MEMBER)
        released_answers = _load_jsonl_member(bundle, _ANSWER_MEMBER)
    if len(prompts) != len(released_answers) or len(prompts) != 5_000:
        raise ValueError("released TextVQA fixtures must contain 5,000 rows")

    local_scores = []
    upstream_scores = []
    normalization_mismatches: list[dict[str, Any]] = []
    score_mismatches: list[dict[str, Any]] = []
    for result in released_answers:
        question = str(result["prompt"]).split("\n", 1)[0].lower()
        annotation = annotations_by_image_question[(str(result["question_id"]), question)]
        prediction = str(result["text"])
        answers = tuple(str(answer) for answer in annotation["answers"])
        local_prediction = normalize_textvqa_answer(prediction)
        upstream_prediction = reference_processor(prediction)
        local_score = _score(normalize_textvqa_answer, prediction, answers)
        upstream_score = _score(reference_processor, prediction, answers)
        local_scores.append(local_score)
        upstream_scores.append(upstream_score)
        if local_prediction != upstream_prediction:
            normalization_mismatches.append(
                {
                    "question_id": annotation["question_id"],
                    "prediction": prediction,
                    "local": local_prediction,
                    "upstream": upstream_prediction,
                }
            )
        if local_score != upstream_score:
            score_mismatches.append(
                {
                    "question_id": annotation["question_id"],
                    "prediction": prediction,
                    "local": local_score,
                    "upstream": upstream_score,
                }
            )

    examples = load_examples(args.examples)
    prompt_index = {
        (str(row["question_id"]), str(row["text"]).split("\n", 1)[0].lower()): row
        for row in prompts
    }
    prefix = f"{VICUNA_V1_SYSTEM} USER: "
    suffix = " ASSISTANT:"
    prompt_mismatches: list[str] = []
    reference_mismatches: list[str] = []
    image_mismatches: list[str] = []
    for example in examples:
        annotation = annotations_by_question[example.id]
        if not example.prompt.startswith(prefix) or not example.prompt.endswith(suffix):
            prompt_mismatches.append(example.id)
            continue
        user_prompt = example.prompt[len(prefix) : -len(suffix)]
        fixture = prompt_index.get(
            (str(annotation["image_id"]), str(annotation["question"]).lower())
        )
        if fixture is None or user_prompt != "<image>\n" + str(fixture["text"]):
            prompt_mismatches.append(example.id)
        if example.references != tuple(str(answer) for answer in annotation["answers"]):
            reference_mismatches.append(example.id)
        if (
            len(example.images) != 1
            or not example.images[0].is_file()
            or str(example.metadata.get("image_id")) != str(annotation["image_id"])
        ):
            image_mismatches.append(example.id)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    count_contract = (
        len(annotations) == len(prompts) == len(released_answers) == len(examples) == 5_000
        and manifest.get("sample_count") == 5_000
        and len(manifest.get("images", ())) == 3_166
        and manifest.get("image_integrity", {}).get("unique_images") == 3_166
        and manifest.get("image_integrity", {}).get("decoded_images") == 3_166
        and manifest.get("image_integrity", {}).get("passed") is True
    )
    mismatch_count = (
        len(normalization_mismatches)
        + len(score_mismatches)
        + len(prompt_mismatches)
        + len(reference_mismatches)
        + len(image_mismatches)
    )
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "textvqa",
        "verifier_sha256": sha256_file(Path(__file__)),
        "upstream_revision": args.upstream_revision,
        "upstream_evaluator": str(args.upstream_evaluator.resolve()),
        "upstream_evaluator_sha256": sha256_file(args.upstream_evaluator),
        "annotations_sha256": sha256_file(args.annotations),
        "llava_archive_sha256": sha256_file(args.llava_archive),
        "examples_sha256": sha256_file(args.examples),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(examples),
        "unique_image_count": len(manifest.get("images", ())),
        "local_released_accuracy": sum(local_scores) / len(local_scores),
        "upstream_released_accuracy": sum(upstream_scores) / len(upstream_scores),
        "normalization_mismatch_count": len(normalization_mismatches),
        "score_mismatch_count": len(score_mismatches),
        "prompt_mismatch_count": len(prompt_mismatches),
        "reference_mismatch_count": len(reference_mismatches),
        "image_binding_mismatch_count": len(image_mismatches),
        "mismatch_examples": {
            "normalization": normalization_mismatches[:20],
            "score": score_mismatches[:20],
            "prompts": prompt_mismatches[:20],
            "references": reference_mismatches[:20],
            "images": image_mismatches[:20],
        },
        "count_contract_passed": count_contract,
        "passed": mismatch_count == 0 and count_contract,
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"TextVQA golden failed with {mismatch_count} mismatches")
    print(
        "TextVQA golden passed: "
        f"{len(examples)} samples, {len(manifest['images'])} images, "
        f"released accuracy {payload['local_released_accuracy']:.5f}"
    )


if __name__ == "__main__":
    main()
