#!/usr/bin/env python3
"""Golden-test VQAv2 conversion and scoring against its pinned evaluator."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.identifiers import benchmark_protocol_id
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples, prepare_vqav2
from invllava.eval.protocols import build_protocol
from invllava.eval.protocols.evalai import normalize_evalai_answer

_EXPECTED_ROWS = 214_354
_EXPECTED_IMAGES = 40_504
_PYTHON2_PRINT = re.compile(r'^(\s*)print\s+("[^"]*")\s*$', re.MULTILINE)


def _official_evaluator(path: Path) -> type[Any]:
    """Load the upstream Python 2 class after its two print statements are modernized."""

    source = _PYTHON2_PRINT.sub(r"\1print(\2)", path.read_text(encoding="utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source, filename=str(path))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQAEval"
    ]
    if len(classes) != 1:
        raise ValueError("official VQAv2 source must define one VQAEval class")
    namespace: dict[str, Any] = {"re": re, "sys": sys}
    module = ast.Module(body=classes, type_ignores=[])
    ast.fix_missing_locations(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["VQAEval"]


def _candidate_panels(examples: list[Any]) -> Iterator[tuple[str, dict[str, str]]]:
    stress = ("Dont", "The TWO, cats!", "1,000.5", "cat's", "Id've")
    for panel_name in (
        "first_reference",
        "last_reference",
        "uppercase_punctuation",
        "article_punctuation",
        "normalization_stress",
    ):
        predictions: dict[str, str] = {}
        for index, example in enumerate(examples):
            first = example.references[0]
            if panel_name == "first_reference":
                prediction = first
            elif panel_name == "last_reference":
                prediction = example.references[-1]
            elif panel_name == "uppercase_punctuation":
                prediction = first.upper() + "!"
            elif panel_name == "article_punctuation":
                prediction = "The " + first + "!"
            else:
                prediction = stress[index % len(stress)]
            predictions[example.id] = prediction
        yield panel_name, predictions


def _official_score(
    evaluator_class: type[Any],
    examples: list[Any],
    predictions: dict[str, str],
) -> tuple[dict[str, float], dict[str, Any]]:
    question_ids = [int(example.id) for example in examples]
    ground_truth = SimpleNamespace(
        qa={
            int(example.id): {
                "answers": [
                    {"answer_id": index, "answer": answer}
                    for index, answer in enumerate(example.references)
                ],
                "answer_type": example.metadata["answer_type"],
                "question_type": example.metadata["question_type"],
            }
            for example in examples
        },
        getQuesIds=lambda: question_ids,
    )
    result = SimpleNamespace(
        qa={int(sample_id): {"answer": prediction} for sample_id, prediction in predictions.items()}
    )
    evaluator = evaluator_class(ground_truth, result, n=10)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return (
        {str(sample_id): float(value) / 100.0 for sample_id, value in evaluator.evalQA.items()},
        {
            "overall": float(evaluator.accuracy["overall"]) / 100.0,
            "answer_type_accuracy": {
                key: float(value) / 100.0
                for key, value in evaluator.accuracy["perAnswerType"].items()
            },
            "question_type_accuracy": {
                key: float(value) / 100.0
                for key, value in evaluator.accuracy["perQuestionType"].items()
            },
        },
    )


def _normalization_probes(evaluator_class: type[Any]) -> dict[str, dict[str, str]]:
    empty = SimpleNamespace(qa={}, getQuesIds=lambda: [])
    evaluator = evaluator_class(empty, empty, n=10)
    probes = (
        "Dont",
        "The TWO, cats!",
        "1,000.5",
        "cat's",
        "Id've",
        "A red-and-blue bus.",
    )
    return {
        value: {
            "native": normalize_evalai_answer(value),
            "official": evaluator.processDigitArticle(evaluator.processPunctuation(value)),
        }
        for value in probes
    }


def _rounded_equal(left: float, right: float) -> bool:
    # The upstream evaluator rounds percentage values to ``n=10`` and only the
    # golden adapter converts them back to fractions.
    return round(100.0 * float(left), 10) == round(100.0 * float(right), 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    spec = BenchmarkSpec.model_validate(yaml.safe_load(args.benchmark.read_text()))
    expected = prepare_vqav2(
        args.questions,
        Path("images"),
        coco_split="val2014",
        spec=spec,
        annotations_path=args.annotations,
    )
    examples = load_examples(args.examples)
    if len(expected) != _EXPECTED_ROWS or len(examples) != _EXPECTED_ROWS:
        raise ValueError(f"VQAv2 validation must contain {_EXPECTED_ROWS} questions")
    expected_by_id = {example.id: example for example in expected}
    examples_by_id = {example.id: example for example in examples}
    if expected_by_id.keys() != examples_by_id.keys():
        raise ValueError("materialized VQAv2 sample IDs differ from the official release")
    structure_mismatches: list[str] = []
    for sample_id, wanted in expected_by_id.items():
        observed = examples_by_id[sample_id]
        if (
            observed.prompt != wanted.prompt
            or observed.references != wanted.references
            or observed.metadata != wanted.metadata
            or tuple(path.name for path in observed.images)
            != tuple(path.name for path in wanted.images)
        ):
            structure_mismatches.append(sample_id)
    del expected, expected_by_id, examples_by_id

    manifest = json.loads(args.manifest.read_text())
    image_names = {example.images[0].name for example in examples}
    manifest_images = {Path(item["file"]).name for item in manifest.get("images", [])}
    manifest_contract = (
        manifest.get("sample_count") == _EXPECTED_ROWS
        and manifest.get("unique_image_count") == _EXPECTED_IMAGES
        and manifest.get("image_integrity", {}).get("unique_images") == _EXPECTED_IMAGES
        and manifest.get("image_integrity", {}).get("passed") is True
        and len(image_names) == _EXPECTED_IMAGES
        and manifest_images == image_names
    )

    evaluator_class = _official_evaluator(args.official_evaluator)
    protocol = build_protocol(spec)
    mismatch_examples: list[dict[str, Any]] = []
    panel_results: dict[str, Any] = {}
    for panel_name, predictions in _candidate_panels(examples):
        official_items, official_summary = _official_score(evaluator_class, examples, predictions)
        native = protocol.score(predictions, examples)
        item_mismatches = [
            sample_id
            for sample_id, value in native.details["per_item"].items()
            if not _rounded_equal(value, official_items[sample_id])
        ]
        summary_mismatches: list[dict[str, Any]] = []
        if not _rounded_equal(native.value, official_summary["overall"]):
            summary_mismatches.append(
                {
                    "metric": "overall",
                    "native": native.value,
                    "official": official_summary["overall"],
                }
            )
        for detail_name in ("answer_type_accuracy", "question_type_accuracy"):
            native_detail = native.details[detail_name]
            official_detail = official_summary[detail_name]
            if native_detail.keys() != official_detail.keys():
                summary_mismatches.append(
                    {
                        "metric": f"{detail_name}.keys",
                        "native": sorted(native_detail),
                        "official": sorted(official_detail),
                    }
                )
                continue
            summary_mismatches.extend(
                {
                    "metric": f"{detail_name}.{key}",
                    "native": value,
                    "official": official_detail[key],
                }
                for key, value in native_detail.items()
                if not _rounded_equal(value, official_detail[key])
            )
        if item_mismatches or summary_mismatches:
            mismatch_examples.append(
                {
                    "panel": panel_name,
                    "items": item_mismatches[:20],
                    "summary": summary_mismatches[:20],
                }
            )
        panel_results[panel_name] = {
            "native": native.value,
            "official": official_summary["overall"],
            "item_mismatches": len(item_mismatches),
            "summary_mismatches": len(summary_mismatches),
        }

    normalization_probes = _normalization_probes(evaluator_class)
    normalization_mismatches = [
        value
        for value, outputs in normalization_probes.items()
        if outputs["native"] != outputs["official"]
    ]
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": spec.id,
        "protocol_id": benchmark_protocol_id(spec),
        "verifier_sha256": sha256_file(Path(__file__)),
        "upstream_revision": args.upstream_revision,
        "official_evaluator_sha256": sha256_file(args.official_evaluator),
        "questions_sha256": sha256_file(args.questions),
        "annotations_sha256": sha256_file(args.annotations),
        "benchmark_config_sha256": sha256_file(args.benchmark),
        "examples_sha256": sha256_file(args.examples),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(examples),
        "unique_image_count": len(image_names),
        "structure_mismatch_count": len(structure_mismatches),
        "manifest_contract_passed": manifest_contract,
        "normalization_probes": normalization_probes,
        "normalization_mismatches": normalization_mismatches,
        "panels": panel_results,
        "mismatch_examples": mismatch_examples,
        "passed": (
            not structure_mismatches
            and manifest_contract
            and not normalization_mismatches
            and not mismatch_examples
        ),
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError("VQAv2 golden failed")
    print(f"VQAv2 golden passed: {len(examples)} rows across {len(panel_results)} panels")


if __name__ == "__main__":
    main()
