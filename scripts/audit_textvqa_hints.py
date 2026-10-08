#!/usr/bin/env python3
"""Audit answer availability and selection in completed OCR-assisted TextVQA runs.

This is a post-result diagnostic, not an alternative benchmark scorer. It uses
the rendered OCR hint string, so span matches need not be one original OCR token
or visually adjacent text. No model outputs are edited or selected for scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256
from invllava.eval.protocols.evalai import normalize_textvqa_answer
from invllava.eval.protocols.vqa import TextVQAProtocol
from invllava.eval.types import EvaluationExample
from scripts.audit_prediction_responses import audit as validate_predictions


def contains_span(text: str, phrase: str) -> bool:
    """Match nonempty normalized word sequences, never partial words."""
    return bool(phrase) and f" {phrase} " in f" {text} "


def hint_text(prompt: str) -> str:
    marker = "\nReference OCR token: "
    if prompt.count(marker) != 1:
        raise ValueError("expected one official reference-OCR prompt line")
    line, separator, _ = prompt.split(marker, 1)[1].partition("\n")
    if not separator:
        raise ValueError("reference-OCR line must end before the answer instruction")
    return normalize_textvqa_answer(line)


def diagnose(records: list[dict], score: dict) -> dict:
    validate_predictions(records, score)
    examples = [
        EvaluationExample(row["sample_id"], row["prompt"], (), tuple(row["references"]))
        for row in records
    ]
    rescored = TextVQAProtocol(score["protocol_id"]).score(
        {row["sample_id"]: row["prediction"] for row in records}, examples
    )
    if not math.isclose(rescored.value, score["value"], rel_tol=0, abs_tol=1e-12):
        raise ValueError("stored aggregate differs from official TextVQA rescoring")
    for sample_id, value in rescored.details["per_item"].items():
        if not math.isclose(
            value, score["details"]["per_item"][sample_id], rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("stored per-item score differs from official TextVQA rescoring")
    items = {}
    for row in records:
        hints = hint_text(row["prompt"])
        answer = normalize_textvqa_answer(row["prediction"])
        counts = Counter(normalize_textvqa_answer(ref) for ref in row["references"])
        # At least two of ten annotators support these answers (consensus >= .6).
        supported = sorted(ref for ref, count in counts.items() if ref and count >= 2)
        items[row["sample_id"]] = {
            "score": rescored.details["per_item"][row["sample_id"]],
            "supported_reference_in_hints": any(contains_span(hints, ref) for ref in supported),
            "prediction_in_hints": contains_span(hints, answer),
            "longer_prediction_contains_supported_reference": any(
                answer != ref and contains_span(answer, ref) for ref in supported
            ),
            "normalized_answer_words": len(answer.split()),
            "normalized_hint_words": len(hints.split()),
        }
    return {"count": len(items), "value": rescored.value, "per_item": items}


def compare(parent: dict, treatment: dict) -> dict:
    if parent["per_item"].keys() != treatment["per_item"].keys():
        raise ValueError("diagnostic sample IDs differ")
    groups = {}
    for available in (False, True):
        pairs = [
            (before, treatment["per_item"][sample_id])
            for sample_id, before in parent["per_item"].items()
            if before["supported_reference_in_hints"] == available
        ]
        if any(
            a["supported_reference_in_hints"] != b["supported_reference_in_hints"] for a, b in pairs
        ):
            raise ValueError("hint/reference membership changed between models")
        count = len(pairs)
        groups["supported_answer_present" if available else "supported_answer_absent"] = {
            "count": count,
            "parent_accuracy": sum(a["score"] for a, _ in pairs) / count if count else None,
            "treatment_accuracy": sum(b["score"] for _, b in pairs) / count if count else None,
            "delta_points": 100 * sum(b["score"] - a["score"] for a, b in pairs) / count
            if count
            else None,
            "lost_score_sum": sum(max(0, a["score"] - b["score"]) for a, b in pairs),
            "gained_score_sum": sum(max(0, b["score"] - a["score"]) for a, b in pairs),
            "declining_items": sum(b["score"] < a["score"] for a, b in pairs),
            "declining_items_with_treatment_hint_span": sum(
                b["score"] < a["score"] and b["prediction_in_hints"] for a, b in pairs
            ),
            "declining_items_with_longer_reference_containing_answer": sum(
                b["score"] < a["score"] and b["longer_prediction_contains_supported_reference"]
                for a, b in pairs
            ),
        }
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "PREDICTIONS", "SCORE"),
    )
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results, sources, identity = {}, {}, None
    for label, prediction_name, score_name in args.series:
        if label in results:
            raise ValueError("duplicate series label")
        predictions, score_path = Path(prediction_name), Path(score_name)
        score = json.loads(score_path.read_text())
        if sha256_file(predictions) != score["predictions_sha256"]:
            raise ValueError("score does not bind predictions")
        rows = [json.loads(line) for line in predictions.read_text().splitlines() if line]
        signature = {
            row["sample_id"]: {
                key: row[key]
                for key in ("prompt", "references", "image_ids", "protocol_id", "generation")
            }
            for row in rows
        }
        if identity is not None and signature != identity:
            raise ValueError("model inputs, references, or generation policies differ")
        identity = signature
        results[label] = diagnose(rows, score)
        sources[label] = {
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "score": str(score_path),
            "score_sha256": sha256_file(score_path),
            "checkpoint_id": score["checkpoint_id"],
            "protocol_id": score["protocol_id"],
        }
    if args.parent not in results:
        raise ValueError("parent must name a supplied series")
    atomic_write_json(
        args.output,
        {
            "series": results,
            "sources": sources,
            "parent": args.parent,
            "contrasts": {
                label: compare(results[args.parent], result)
                for label, result in results.items()
                if label != args.parent
            },
            "script_sha256": sha256_file(__file__),
            "execution_source_sha256": execution_source_sha256(Path(__file__).resolve().parents[1])[
                0
            ],
            "scope": "Exploratory descriptive audit, conditional on completed checkpoints. "
            "Official scores are unchanged. Reference support means at least two annotators. "
            "Normalized spans use the rendered hint order and can cross OCR-token boundaries; "
            "they do not prove copying, visual recognition, or causal mechanisms. "
            "Reference-containing longer answers are diagnostic candidates, "
            "not corrected predictions. "
            "No confidence intervals or independent confirmation are claimed.",
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
