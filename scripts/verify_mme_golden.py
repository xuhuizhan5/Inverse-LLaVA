#!/usr/bin/env python3
"""Verify MME prompts, references, image bindings, parsing, and aggregation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.mme import MMEProtocol, extract_mme_answer
from invllava.prompting import format_vicuna_v1_user_prompt

_QUESTION_MEMBER = "MME/llava_mme.jsonl"
_ANSWER_MEMBER = "MME/answers/llava-v1.5-13b.jsonl"
_RELEASE_SUFFIX = "Please answer yes or no."
_LLAVA_SUFFIX = "Answer the question using a single word or phrase."


def _load_upstream(path: Path) -> Any:
    logger_module = types.ModuleType("loguru")
    logger_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules.setdefault("loguru", logger_module)
    spec = importlib.util.spec_from_file_location("pinned_mme_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned MME utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_jsonl_member(bundle: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in bundle.read(member).decode("utf-8").splitlines()]


def _annotation_members(bundle: zipfile.ZipFile, categories: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for info in bundle.infolist():
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".txt":
            continue
        parts = PurePosixPath(info.filename).parts
        positions = [index for index, value in enumerate(parts) if value in categories]
        if len(positions) != 1:
            continue
        key = f"{parts[positions[0]]}/{parts[-1]}"
        if key in result:
            raise ValueError(f"duplicate MME annotation key {key}")
        result[key] = info.filename
    return result


def _pairs(bundle: zipfile.ZipFile, member: str) -> list[tuple[str, str]]:
    result = []
    for line in bundle.read(member).decode("utf-8").splitlines():
        question, answer = line.split("\t")
        result.append((question, answer))
    if len(result) != 2:
        raise ValueError(f"MME annotation {member} does not contain two questions")
    return result


def _canonical_without_project_code(question: str) -> str:
    value = question.strip()
    if not value.endswith(_RELEASE_SUFFIX):
        raise ValueError("MME annotation lacks its released suffix")
    return f"{value[: -len(_RELEASE_SUFFIX)].rstrip()}\n{_LLAVA_SUFFIX}"


def _manifest_images(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {Path(row["file"]).name: row for row in manifest["images"]}, manifest


def _candidates() -> tuple[str, ...]:
    return (
        "yes",
        "no",
        "Yes.",
        "No.",
        "y",
        "n",
        "Yes, because it is visible.",
        "No, because it is absent.",
        "yesman",
        "maybe",
        "I think yes",
        "",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--llava-archive", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument("--perception-examples", type=Path, required=True)
    parser.add_argument("--perception-manifest", type=Path, required=True)
    parser.add_argument("--cognition-examples", type=Path, required=True)
    parser.add_argument("--cognition-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    upstream = _load_upstream(args.upstream_utils)
    perception = load_examples(args.perception_examples)
    cognition = load_examples(args.cognition_examples)
    examples = perception + cognition
    example_index = {example.id: example for example in examples}
    if len(example_index) != 2_374:
        raise ValueError("prepared MME domains must contain 2,374 unique questions")
    perception_images, perception_manifest = _manifest_images(args.perception_manifest)
    cognition_images, cognition_manifest = _manifest_images(args.cognition_manifest)

    with zipfile.ZipFile(args.llava_archive) as llava_bundle:
        fixture = _load_jsonl_member(llava_bundle, _QUESTION_MEMBER)
        released_answers = _load_jsonl_member(llava_bundle, _ANSWER_MEMBER)
    with zipfile.ZipFile(args.data_archive) as data_bundle:
        categories = set(
            upstream.eval_type_dict["Perception"] + upstream.eval_type_dict["Cognition"]
        )
        annotation_index = _annotation_members(data_bundle, categories)
        annotation_cache: dict[str, list[tuple[str, str]]] = {}
        occurrences: Counter[str] = Counter()
        prompt_mismatches: list[str] = []
        reference_mismatches: list[str] = []
        image_mismatches: list[str] = []
        expected: dict[str, dict[str, Any]] = {}
        for row in fixture:
            group_id = str(row["question_id"])
            pair_index = occurrences[group_id]
            occurrences[group_id] += 1
            sample_id = f"{group_id}/{pair_index}"
            category = str(row["category"])
            annotation_key = f"{category}/{PurePosixPath(group_id).with_suffix('.txt').name}"
            if group_id not in annotation_cache:
                annotation_cache[group_id] = _pairs(
                    data_bundle,
                    annotation_index[annotation_key],
                )
            question, reference = annotation_cache[group_id][pair_index]
            expected_text = _canonical_without_project_code(question)
            example = example_index[sample_id]
            if str(row["text"]) != expected_text or example.prompt != format_vicuna_v1_user_prompt(
                f"<image>\n{expected_text}"
            ):
                prompt_mismatches.append(sample_id)
            if example.references != (reference,):
                reference_mismatches.append(sample_id)
            manifest_images = (
                perception_images
                if category in upstream.eval_type_dict["Perception"]
                else cognition_images
            )
            record = manifest_images.get(example.images[0].name)
            source_member = PurePosixPath(str(record["source_member"])) if record else None
            source_key = (
                f"{category}/{source_member.name}"
                if source_member is not None and category in source_member.parts
                else None
            )
            expected_image_key = f"{category}/{PurePosixPath(group_id).name}"
            if (
                len(example.images) != 1
                or not example.images[0].is_file()
                or record is None
                or source_key != expected_image_key
            ):
                image_mismatches.append(sample_id)
            expected[sample_id] = {"row": row, "answer": reference}

    group_mismatches = [group for group, count in occurrences.items() if count != 2]
    parser_mismatches: list[dict[str, Any]] = []
    comparisons = 0
    for prediction in _candidates():
        upstream_value = upstream.parse_pred_ans(prediction)
        upstream_value = None if upstream_value == "other" else upstream_value
        local_value = extract_mme_answer(prediction)
        comparisons += 1
        if upstream_value != local_value:
            parser_mismatches.append(
                {"prediction": prediction, "upstream": upstream_value, "local": local_value}
            )

    answer_occurrences: Counter[str] = Counter()
    predictions: dict[str, str] = {}
    upstream_results: dict[str, list[dict[str, Any]]] = {
        "perception": [],
        "cognition": [],
    }
    released_prompt_mismatches: list[str] = []
    for answer in released_answers:
        group_id = str(answer["question_id"])
        pair_index = answer_occurrences[group_id]
        answer_occurrences[group_id] += 1
        sample_id = f"{group_id}/{pair_index}"
        expected_row = expected[sample_id]
        row = expected_row["row"]
        if str(answer["prompt"]) != str(row["text"]):
            released_prompt_mismatches.append(sample_id)
        prediction = str(answer["text"])
        predictions[sample_id] = prediction
        document = {
            "question_id": group_id,
            "category": row["category"],
            "answer": expected_row["answer"],
        }
        result = upstream.mme_process_results(document, [prediction])
        domain = (
            "perception"
            if row["category"] in upstream.eval_type_dict["Perception"]
            else "cognition"
        )
        upstream_results[domain].append(result[f"mme_{domain}_score"])

    aggregate_mismatches: list[dict[str, Any]] = []
    released_scores: dict[str, dict[str, Any]] = {}
    for domain, domain_examples in (("perception", perception), ("cognition", cognition)):
        local = MMEProtocol("golden", domain=domain).score(predictions, domain_examples)
        upstream_total = upstream.mme_aggregate_results(upstream_results[domain])
        if not math.isclose(local.value, upstream_total, rel_tol=0.0, abs_tol=1e-12):
            aggregate_mismatches.append(
                {"scope": domain, "upstream": upstream_total, "local": local.value}
            )
        upstream_categories = {}
        for category in upstream.eval_type_dict[domain.capitalize()]:
            rows = [row for row in upstream_results[domain] if row["category"] == category]
            upstream_categories[category] = upstream.mme_aggregate_results(rows)
            if not math.isclose(
                local.details["category_scores"][category],
                upstream_categories[category],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                aggregate_mismatches.append(
                    {
                        "scope": category,
                        "upstream": upstream_categories[category],
                        "local": local.details["category_scores"][category],
                    }
                )
        released_scores[domain] = {
            "upstream": upstream_total,
            "local": local.value,
            "category_scores": upstream_categories,
        }

    mismatches = (
        len(prompt_mismatches)
        + len(reference_mismatches)
        + len(image_mismatches)
        + len(group_mismatches)
        + len(parser_mismatches)
        + len(released_prompt_mismatches)
        + len(aggregate_mismatches)
    )
    count_contract = (
        len(fixture) == len(released_answers) == len(examples) == 2_374
        and len(perception) == perception_manifest["sample_count"] == 2_114
        and len(cognition) == cognition_manifest["sample_count"] == 260
        and perception_manifest["image_integrity"]["unique_images"] == 1_057
        and cognition_manifest["image_integrity"]["unique_images"] == 130
        and perception_manifest["image_integrity"]["passed"]
        and cognition_manifest["image_integrity"]["passed"]
    )
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "mme",
        "verifier_sha256": sha256_file(Path(__file__)),
        "upstream_revision": args.upstream_revision,
        "upstream_utils": str(args.upstream_utils.resolve()),
        "upstream_utils_sha256": sha256_file(args.upstream_utils),
        "llava_archive_sha256": sha256_file(args.llava_archive),
        "data_archive_sha256": sha256_file(args.data_archive),
        "sample_count": len(examples),
        "parser_comparison_count": comparisons,
        "prompt_mismatch_count": len(prompt_mismatches),
        "reference_mismatch_count": len(reference_mismatches),
        "image_binding_mismatch_count": len(image_mismatches),
        "group_mismatch_count": len(group_mismatches),
        "parser_mismatch_count": len(parser_mismatches),
        "released_prompt_mismatch_count": len(released_prompt_mismatches),
        "aggregate_mismatch_count": len(aggregate_mismatches),
        "mismatch_examples": {
            "prompts": prompt_mismatches[:20],
            "references": reference_mismatches[:20],
            "images": image_mismatches[:20],
            "groups": group_mismatches[:20],
            "parser": parser_mismatches[:20],
            "released_prompts": released_prompt_mismatches[:20],
            "aggregate": aggregate_mismatches[:20],
        },
        "released_llava_1_5_13b_scores": released_scores,
        "count_contract_passed": count_contract,
        "passed": mismatches == 0 and count_contract,
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"MME golden failed with {mismatches} mismatches")
    print(f"MME golden passed: {len(examples)} samples and {comparisons} parser comparisons")


if __name__ == "__main__":
    main()
