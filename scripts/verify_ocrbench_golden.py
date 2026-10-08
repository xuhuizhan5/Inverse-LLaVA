#!/usr/bin/env python3
"""Differentially verify OCRBench scoring against a pinned lmms-eval utility."""

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
from invllava.eval.protocols.ocrbench import OCRBenchProtocol


def _load_upstream(path: Path) -> Any:
    logger_module = types.ModuleType("loguru")
    logger_module.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules.setdefault("loguru", logger_module)
    for name in (
        "lmms_eval",
        "lmms_eval.tasks",
        "lmms_eval.tasks._task_utils",
        "lmms_eval.tasks._task_utils.file_utils",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    file_utils = sys.modules["lmms_eval.tasks._task_utils.file_utils"]
    file_utils.generate_submission_file = lambda *_args, **_kwargs: "unused"
    spec = importlib.util.spec_from_file_location("pinned_ocrbench_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned OCRBench utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fullwidth_ascii(value: str) -> str:
    return "".join(
        chr(ord(character) + 0xFEE0) if 0x21 <= ord(character) <= 0x7E else character
        for character in value
    ).replace(" ", "\u3000")


def _candidates(references: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = ["∅∅∅"]
    for reference in references:
        values.extend(
            (
                reference,
                f"Answer: {reference}",
                reference.swapcase(),
                reference.replace(" ", ""),
                _fullwidth_ascii(reference),
            )
        )
    return tuple(dict.fromkeys(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    upstream = _load_upstream(args.upstream_utils)
    local = OCRBenchProtocol()
    examples = load_examples(args.examples)
    comparisons = 0
    mismatches: list[dict[str, Any]] = []
    for example in examples:
        dataset = str(example.metadata["dataset"])
        question_type = str(example.metadata["question_type"])
        answer: str | list[str] = (
            example.references[0] if len(example.references) == 1 else list(example.references)
        )
        document = {
            "answer": answer,
            "dataset": dataset,
            "question_type": question_type,
        }
        for prediction in _candidates(example.references):
            upstream_score = int(
                upstream.ocrbench_process_results(document, [prediction])["ocrbench_accuracy"][
                    "score"
                ]
            )
            local_score = local._correct(prediction, example.references, dataset)
            comparisons += 1
            if upstream_score != local_score:
                mismatches.append(
                    {
                        "sample_id": example.id,
                        "dataset": dataset,
                        "prediction": prediction,
                        "upstream": upstream_score,
                        "local": local_score,
                    }
                )
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "ocrbench",
        "upstream_revision": args.upstream_revision,
        "upstream_utils": str(args.upstream_utils.resolve()),
        "upstream_utils_sha256": sha256_file(args.upstream_utils),
        "examples": str(args.examples.resolve()),
        "examples_sha256": sha256_file(args.examples),
        "sample_count": len(examples),
        "comparison_count": comparisons,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:100],
        "passed": not mismatches,
    }
    atomic_write_json(args.output, payload)
    if mismatches:
        raise RuntimeError(f"OCRBench scorer golden found {len(mismatches)} mismatches")
    print(f"OCRBench scorer golden passed: {comparisons} comparisons")


if __name__ == "__main__":
    main()
