#!/usr/bin/env python3
"""Differentially verify AI2D prompts, targets, filtering, and scoring."""

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
from invllava.eval.protocols.ai2d import (
    AI2DProtocol,
    filter_ai2d_response,
    normalize_ai2d_exact,
)
from invllava.prompting import format_vicuna_v1_user_prompt

_DEFAULT_KWARGS = {
    "prompt_format": "mcq",
    "pre_prompt": "",
    "post_prompt": "\nAnswer with the option's letter from the given choices directly.",
}


class _ExtendedRegexFilter:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


def _load_upstream(path: Path) -> Any:
    lmms_eval = types.ModuleType("lmms_eval")
    filters = types.ModuleType("lmms_eval.filters")
    extraction = types.ModuleType("lmms_eval.filters.extraction")
    extraction.ExtendedRegexFilter = _ExtendedRegexFilter  # type: ignore[attr-defined]
    sys.modules.setdefault("lmms_eval", lmms_eval)
    sys.modules.setdefault("lmms_eval.filters", filters)
    sys.modules.setdefault("lmms_eval.filters.extraction", extraction)
    spec = importlib.util.spec_from_file_location("pinned_ai2d_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned AI2D utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidates(answer: str) -> tuple[str, ...]:
    wrong = "A" if answer != "A" else "B"
    return (
        answer,
        answer.lower(),
        f"{answer}.",
        f"{answer}. option text",
        f"The answer is {answer}.",
        wrong,
        "",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-utils", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from datasets import load_dataset

    upstream = _load_upstream(args.upstream_utils)
    examples = load_examples(args.examples)
    dataset = load_dataset(
        args.dataset,
        split=args.split,
        revision=args.dataset_revision,
        cache_dir=str(args.cache_dir),
    )
    if len(examples) != len(dataset):
        raise ValueError("AI2D examples and pinned dataset have different sizes")

    response_filter = upstream.MultiChoiceRegexFilter()
    local = AI2DProtocol(args.upstream_revision)
    mismatches: list[dict[str, object]] = []
    comparisons = 0
    predictions: dict[str, str] = {}
    for index, (example, row) in enumerate(zip(examples, dataset, strict=True)):
        row = dict(row)
        expected_id = str(row.get("id", index))
        upstream_prompt = upstream.ai2d_doc_to_text(row, _DEFAULT_KWARGS)
        expected_prompt = format_vicuna_v1_user_prompt(f"<image>\n{upstream_prompt}")
        upstream_target = upstream.ai2d_doc_to_target(row, "mcq")
        if example.id != expected_id or example.prompt != expected_prompt:
            mismatches.append({"sample_id": example.id, "scope": "prompt_or_id"})
        if example.references != (upstream_target,):
            mismatches.append({"sample_id": example.id, "scope": "target"})
        for prediction in _candidates(upstream_target):
            upstream_filtered = response_filter.apply([[prediction]], [row])[0]
            local_filtered = filter_ai2d_response(prediction)
            upstream_score = int(
                normalize_ai2d_exact(upstream_filtered) == normalize_ai2d_exact(upstream_target)
            )
            local_score = local.score({example.id: prediction}, [example]).value
            comparisons += 1
            if upstream_filtered != local_filtered or upstream_score != local_score:
                mismatches.append(
                    {
                        "sample_id": example.id,
                        "prediction": prediction,
                        "upstream_filtered": upstream_filtered,
                        "local_filtered": local_filtered,
                        "upstream": upstream_score,
                        "local": local_score,
                    }
                )
        predictions[example.id] = (
            upstream_target if index % 3 else "The answer is " + upstream_target
        )

    local_score = local.score(predictions, examples)
    expected_correct = sum(
        normalize_ai2d_exact(filter_ai2d_response(predictions[example.id]))
        == normalize_ai2d_exact(example.references[0])
        for example in examples
    )
    if local_score.details["correct"] != expected_correct:
        mismatches.append(
            {
                "scope": "aggregate",
                "expected_correct": expected_correct,
                "local_correct": local_score.details["correct"],
            }
        )

    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": "ai2d",
        "upstream_revision": args.upstream_revision,
        "upstream_utils": str(args.upstream_utils.resolve()),
        "upstream_utils_sha256": sha256_file(args.upstream_utils),
        "dataset": args.dataset,
        "dataset_revision": args.dataset_revision,
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
        raise RuntimeError(f"AI2D golden found {len(mismatches)} mismatches")
    print(f"AI2D golden passed: {comparisons} comparisons over {len(examples)} samples")


if __name__ == "__main__":
    main()
