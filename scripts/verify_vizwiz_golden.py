#!/usr/bin/env python3
"""Golden-test VizWiz prompts and scores against its pinned public evaluator."""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples, prepare_vizwiz
from invllava.eval.protocols import build_protocol
from invllava.eval.protocols.vizwiz import normalize_vizwiz_answer
from invllava.eval.vizwiz_data import load_llava_questions, verify_llava_questions

_EXPECTED_ROWS = 8_000


class _CaptionMetricStub:
    def __init__(self, *_: Any) -> None:
        self.eval: dict[str, float] = {}

    def evaluate(self) -> None:
        return None


def _official_evaluator(path: Path) -> type[Any]:
    """Execute the upstream VQAEval class without its unrelated caption imports."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQAEval"
    ]
    if len(classes) != 1:
        raise ValueError("official VizWiz source must define one VQAEval class")
    namespace: dict[str, Any] = {
        "sys": sys,
        "re": re,
        "np": None,
        "average_precision_score": None,
        "f1_score": None,
        "COCOEvalCap": _CaptionMetricStub,
    }
    module = ast.Module(body=classes, type_ignores=[])
    ast.fix_missing_locations(module)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        exec(compile(module, str(path), "exec"), namespace)
    return namespace["VQAEval"]


def _candidate_panels(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    panels: dict[str, dict[str, str]] = {
        "first_reference": {},
        "last_reference": {},
        "uppercase_punctuation": {},
        "article_punctuation": {},
        "normalization_stress": {},
    }
    stress = ("Dont", "The TWO, cats!", "1,000.5", "unanswerable", "wheres")
    for index, row in enumerate(rows):
        sample_id = str(row["image"])
        answers = row["answers"]
        first = str(answers[0]["answer"])
        last = str(answers[-1]["answer"])
        panels["first_reference"][sample_id] = first
        panels["last_reference"][sample_id] = last
        panels["uppercase_punctuation"][sample_id] = first.upper() + "!"
        panels["article_punctuation"][sample_id] = "The " + first + "!"
        panels["normalization_stress"][sample_id] = stress[index % len(stress)]
    return panels


def _official_score(
    evaluator_class: type[Any],
    rows: list[dict[str, Any]],
    predictions: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
    ground_truth = SimpleNamespace(
        imgToQA={str(row["image"]): row for row in rows},
        getImgs=lambda: [str(row["image"]) for row in rows],
    )
    result = SimpleNamespace(
        imgToQA={
            sample_id: {"image": sample_id, "answer": prediction}
            for sample_id, prediction in predictions.items()
        }
    )
    evaluator = evaluator_class(ground_truth, result, n=10)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    normalized = {
        sample_id: evaluator.processDigitArticle(
            evaluator.processPunctuation(prediction.replace("\n", " ").replace("\t", " ").strip())
        )
        for sample_id, prediction in predictions.items()
    }
    return (
        {sample_id: float(value) / 100.0 for sample_id, value in evaluator.evalQA.items()},
        {
            "overall": float(evaluator.accuracy["overall"]) / 100.0,
            **{
                f"answer_type.{key}": float(value) / 100.0
                for key, value in evaluator.accuracy["perAnswerType"].items()
            },
        },
        normalized,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    parser.add_argument("--llava-archive", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if bool(args.examples) != bool(args.manifest):
        raise ValueError("provide --examples and --manifest together")

    spec = BenchmarkSpec.model_validate(yaml.safe_load(args.benchmark.read_text(encoding="utf-8")))
    rows = json.loads(args.annotations.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != _EXPECTED_ROWS:
        raise ValueError("public VizWiz test answers must contain 8,000 rows")
    sources = [source for source in spec.protocol_sources if source.id == "llava-v1.5-eval"]
    if len(sources) != 1 or sha256_file(args.llava_archive) != sources[0].sha256:
        raise ValueError("LLaVA prompt archive does not match its pinned source")
    if sha256_file(args.annotations) != spec.annotations.sha256:
        raise ValueError("VizWiz annotations do not match their pinned source")
    expected = prepare_vizwiz(
        args.annotations,
        Path("images"),
        spec=spec,
        llava_questions=load_llava_questions(args.llava_archive),
    )
    prompt_fixture = verify_llava_questions(expected, args.llava_archive)
    examples = load_examples(args.examples) if args.examples else expected
    expected_by_id = {example.id: example for example in expected}
    examples_by_id = {example.id: example for example in examples}
    structure_mismatches: list[str] = []
    if expected_by_id.keys() != examples_by_id.keys():
        raise ValueError("materialized VizWiz sample IDs differ from the public release")
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

    manifest_contract = True
    manifest_sha256 = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest_sha256 = sha256_file(args.manifest)
        manifest_contract = (
            manifest.get("sample_count") == _EXPECTED_ROWS
            and manifest.get("image_integrity", {}).get("unique_images") == _EXPECTED_ROWS
            and manifest.get("image_integrity", {}).get("passed") is True
        )

    evaluator_class = _official_evaluator(args.official_evaluator)
    protocol = build_protocol(spec)
    mismatch_examples: list[dict[str, Any]] = []
    panel_results: dict[str, Any] = {}
    for panel_name, predictions in _candidate_panels(rows).items():
        official_items, official_summary, official_normalized = _official_score(
            evaluator_class, rows, predictions
        )
        native = protocol.score(predictions, examples)
        native_types = native.details["answer_type_accuracy"]
        item_mismatches = [
            sample_id
            for sample_id, value in native.details["per_item"].items()
            if round(float(value), 10) != round(official_items[sample_id], 10)
        ]
        normalization_mismatches = [
            sample_id
            for sample_id, prediction in predictions.items()
            if normalize_vizwiz_answer(prediction) != official_normalized[sample_id]
        ]
        summary_mismatches = []
        if round(native.value, 10) != round(official_summary["overall"], 10):
            summary_mismatches.append("overall")
        for answer_type, value in native_types.items():
            key = f"answer_type.{answer_type}"
            if round(float(value), 10) != round(official_summary[key], 10):
                summary_mismatches.append(key)
        if item_mismatches or normalization_mismatches or summary_mismatches:
            mismatch_examples.append(
                {
                    "panel": panel_name,
                    "items": item_mismatches[:20],
                    "normalization": normalization_mismatches[:20],
                    "summary": summary_mismatches,
                }
            )
        panel_results[panel_name] = {
            "native": native.value,
            "official": official_summary["overall"],
            "item_mismatches": len(item_mismatches),
            "normalization_mismatches": len(normalization_mismatches),
            "summary_mismatches": len(summary_mismatches),
        }

    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": spec.id,
        "upstream_revision": args.upstream_revision,
        "official_evaluator_sha256": sha256_file(args.official_evaluator),
        "annotations_sha256": sha256_file(args.annotations),
        "benchmark_config_sha256": sha256_file(args.benchmark),
        "examples_sha256": sha256_file(args.examples) if args.examples else None,
        "manifest_sha256": manifest_sha256,
        "sample_count": len(examples),
        "structure_mismatch_count": len(structure_mismatches),
        "manifest_contract_passed": manifest_contract,
        "llava_prompt_fixture": prompt_fixture,
        "panels": panel_results,
        "mismatch_examples": mismatch_examples,
        "passed": not structure_mismatches and manifest_contract and not mismatch_examples,
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError("VizWiz golden failed")
    print(f"VizWiz golden passed: {len(examples)} rows across {len(panel_results)} panels")


if __name__ == "__main__":
    main()
