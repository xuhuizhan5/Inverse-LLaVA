#!/usr/bin/env python3
"""Audit ScienceQA format failures without replacing its official score.

The supplementary diagnostic changes only a standalone lowercase option letter
to uppercase. It does not extract choices from explanations or alter any files
used for benchmark scoring. Apply the same diagnostic to every compared model.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.scienceqa import extract_llava_scienceqa_choice
from scripts.audit_prediction_responses import audit as validate_predictions


def diagnose(records: list[dict], score: dict, examples: list) -> dict:
    validate_predictions(records, score)
    by_id = {example.id: example for example in examples}
    if len(by_id) != len(examples) or set(by_id) != {row["sample_id"] for row in records}:
        raise ValueError("examples and predictions must have the same unique IDs")
    counts = Counter()
    items = {}
    for row in records:
        example = by_id[row["sample_id"]]
        if row["prompt"] != example.prompt or tuple(row["references"]) != example.references:
            raise ValueError("question or reference mismatch")
        if len(example.references) != 1 or not example.choices:
            raise ValueError("one reference and nonempty choices required")
        prediction = row["prediction"]
        choice = extract_llava_scienceqa_choice(prediction, len(example.choices))
        correct = int(choice == example.references[0])
        if correct != score["details"]["per_item"][example.id]:
            raise ValueError("official per-item score mismatch")
        standalone_lower = (
            len(prediction) == 1
            and "a" <= prediction <= "z"
            and ord(prediction) - ord("a") < len(example.choices)
        )
        diagnostic = prediction.upper() if standalone_lower else choice
        diagnostic_correct = int(diagnostic == example.references[0])
        counts["official_correct"] += correct
        counts["official_invalid"] += choice is None
        counts["standalone_lowercase_choices"] += standalone_lower
        counts["lowercase_correct_choices"] += standalone_lower and diagnostic_correct
        counts["case_only_diagnostic_correct"] += diagnostic_correct
        items[example.id] = {
            "official_score": correct,
            "official_choice": choice,
            "standalone_lowercase": standalone_lower,
            "case_only_diagnostic_score": diagnostic_correct,
        }
    if not math.isclose(
        counts["official_correct"] / len(records), score["value"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("official aggregate score mismatch")
    if counts["official_invalid"] != score["details"]["invalid"]:
        raise ValueError("official invalid-answer count mismatch")
    return {
        "count": len(records),
        "counts": dict(counts),
        "per_item": items,
        "official_accuracy": score["value"],
        "case_only_diagnostic_accuracy": counts["case_only_diagnostic_correct"] / len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument(
        "--series",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "PREDICTIONS", "SCORE"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    examples = load_examples(args.examples)
    results, sources, identity = {}, {}, None
    for label, prediction_path, score_path in args.series:
        if label in results:
            raise ValueError("duplicate series label")
        predictions, score_file = Path(prediction_path), Path(score_path)
        score = json.loads(score_file.read_text())
        if score["predictions_sha256"] != sha256_file(predictions):
            raise ValueError("prediction hash mismatch")
        if score["examples_sha256"] != sha256_file(args.examples):
            raise ValueError("example hash mismatch")
        rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
        signature = {
            row["sample_id"]: {
                key: row[key]
                for key in ("prompt", "references", "generation", "image_ids", "protocol_id")
            }
            for row in rows
        }
        if identity is not None and signature != identity:
            raise ValueError("comparison input/protocol policies differ")
        identity = signature
        results[label] = diagnose(rows, score, examples)
        sources[label] = {
            "predictions_sha256": sha256_file(predictions),
            "score_sha256": sha256_file(score_file),
            "checkpoint_id": score["checkpoint_id"],
            "protocol_id": score["protocol_id"],
        }
    atomic_write_json(
        args.output,
        {
            "series": results,
            "sources": sources,
            "examples_sha256": sha256_file(args.examples),
            "script_sha256": sha256_file(__file__),
            "scope": "Post-result format diagnostic. Only standalone lowercase letters "
            "are uppercased for a supplementary count; original outputs, official metrics, "
            "and benchmark protocols remain unchanged. This quantifies a case-format effect, "
            "not recovery through training or general reasoning retention. "
            "No uncertainty or independent confirmation is claimed.",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
