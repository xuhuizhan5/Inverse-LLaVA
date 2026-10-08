#!/usr/bin/env python3
"""Differentially verify MMStar extraction and aggregation against lmms-eval."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.mmstar import MMStarProtocol, extract_mmstar_choice


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream(utils_path: Path, extractor_path: Path) -> Any:
    logger_module = types.ModuleType("loguru")
    logger_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules.setdefault("loguru", logger_module)
    for name in ("lmms_eval", "lmms_eval.tasks", "lmms_eval.tasks._task_utils"):
        sys.modules.setdefault(name, types.ModuleType(name))
    _load_module("lmms_eval.tasks._task_utils.mcq_extract", extractor_path)
    return _load_module("pinned_mmstar_utils", utils_path)


def _candidates(answer: str) -> tuple[str, ...]:
    wrong = chr(ord("A") + ((ord(answer) - ord("A") + 1) % 4))
    return (
        "∅∅∅",
        answer,
        f"({answer})",
        f"The correct answer is ({answer}).",
        f"I choose {answer}",
        f"{wrong}. is tempting, but {answer}. is correct",
        wrong,
    )


def _document(example: Any) -> dict[str, Any]:
    return {
        "index": example.id,
        "answer": example.references[0],
        "category": example.metadata["category"],
        "l2_category": example.metadata["l2_category"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--upstream-extractor", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    upstream = _load_upstream(args.upstream_utils, args.upstream_extractor)
    examples = load_examples(args.examples)
    mismatches: list[dict[str, Any]] = []
    comparisons = 0
    for example in examples:
        answer = example.references[0]
        for prediction in _candidates(answer):
            upstream_score = int(upstream.exact_match(prediction, answer))
            local_score = int(extract_mmstar_choice(prediction) == answer)
            comparisons += 1
            if upstream_score != local_score:
                mismatches.append(
                    {
                        "sample_id": example.id,
                        "prediction": prediction,
                        "upstream": upstream_score,
                        "local": local_score,
                    }
                )

    predictions: dict[str, str] = {}
    upstream_average: list[dict[str, Any]] = []
    upstream_categories: dict[str, list[dict[str, Any]]] = {}
    for index, example in enumerate(examples):
        answer = example.references[0]
        prediction = answer if index % 3 else chr(ord("A") + ((ord(answer) - 64) % 4))
        predictions[example.id] = prediction
        result = upstream.mmstar_process_results(_document(example), [prediction])
        upstream_average.append(result["average"])
        category = str(example.metadata["category"])
        upstream_categories.setdefault(category, []).append(result[category])

    local_score = MMStarProtocol(args.upstream_revision).score(predictions, examples)
    upstream_value = upstream.mmstar_aggregate_results(upstream_average)
    if upstream_value != local_score.value:
        mismatches.append(
            {"scope": "average", "upstream": upstream_value, "local": local_score.value}
        )
    for category, results in sorted(upstream_categories.items()):
        upstream_category = upstream.mmstar_aggregate_results(results)
        local_category = local_score.details["category_scores"][category]
        if upstream_category != local_category:
            mismatches.append(
                {
                    "scope": category,
                    "upstream": upstream_category,
                    "local": local_category,
                }
            )

    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "mmstar",
        "upstream_revision": args.upstream_revision,
        "upstream_utils": str(args.upstream_utils.resolve()),
        "upstream_utils_sha256": sha256_file(args.upstream_utils),
        "upstream_extractor": str(args.upstream_extractor.resolve()),
        "upstream_extractor_sha256": sha256_file(args.upstream_extractor),
        "examples": str(args.examples.resolve()),
        "examples_sha256": sha256_file(args.examples),
        "sample_count": len(examples),
        "comparison_count": comparisons,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "upstream_macro_l2": upstream_value,
        "local_macro_l2": local_score.value,
        "passed": not mismatches,
    }
    atomic_write_json(args.output, payload)
    if mismatches:
        raise RuntimeError(f"MMStar scorer golden found {len(mismatches)} mismatches")
    print(f"MMStar scorer golden passed: {comparisons} extraction comparisons and aggregation")


if __name__ == "__main__":
    main()
