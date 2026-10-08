#!/usr/bin/env python3
"""Describe response lengths and repetition without changing benchmark scores.

Use on complete scored predictions to investigate answer-style changes. Word
counts use whitespace splitting, not model tokens. Repetition can be legitimate
for short-answer tasks; none of these summaries diagnose forgetting by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file


def summarize(records: list[dict]) -> dict:
    responses = [record["prediction"] for record in records]
    words = [len(response.split()) for response in responses]
    counts = Counter(" ".join(response.split()).casefold() for response in responses)
    return {
        "count": len(records),
        "empty_responses": sum(not response.strip() for response in responses),
        "mean_characters": mean(map(len, responses)),
        "mean_whitespace_words": mean(words),
        "median_whitespace_words": median(words),
        "unique_normalized_responses": len(counts),
        "most_common_normalized_responses": [
            {"response": text, "count": count}
            for text, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def audit(records: list[dict], score: dict, *, group_field: str | None = None) -> dict:
    ids = [record["sample_id"] for record in records]
    if not records or len(set(ids)) != len(ids) or len(ids) != score["count"]:
        raise ValueError("unique, complete scored predictions are required")
    items = score["details"]["per_item"]
    if set(ids) != set(items):
        raise ValueError("score and prediction IDs differ")
    if any(not isinstance(record["prediction"], str) for record in records):
        raise ValueError("predictions must be strings")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) for value in items.values()
    ):
        raise ValueError("per-item scores must be finite numbers")
    if any(
        record["checkpoint_id"] != score["checkpoint_id"]
        or record["protocol_id"] != score["protocol_id"]
        for record in records
    ):
        raise ValueError("mixed prediction checkpoint/protocol identity")
    groups = defaultdict(list)
    if group_field:
        for record in records:
            label = record.get("metadata", {}).get(group_field)
            if not isinstance(label, str) or not label:
                raise ValueError(f"missing string metadata field: {group_field}")
            groups[label].append(record)
    return {
        "overall": summarize(records),
        "group_field": group_field,
        "groups": {label: summarize(rows) for label, rows in sorted(groups.items())},
        "score": score["value"],
        "scope": "Descriptive full-endpoint audit. Whitespace words are not tokenizer tokens. "
        "Normalization only collapses whitespace and case for repetition counts. "
        "No predictions or official scores are changed. "
        "Response style alone does not establish causality.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument(
        "--group-field", help="Existing metadata field, e.g. question_type for OCRBench"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = json.loads(args.score.read_text())
    if score["predictions_sha256"] != sha256_file(args.predictions):
        raise ValueError("score does not bind the supplied predictions")
    records = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    result = audit(records, score, group_field=args.group_field)
    atomic_write_json(
        args.output,
        {
            **result,
            "checkpoint_id": score["checkpoint_id"],
            "protocol_id": score["protocol_id"],
            "predictions_sha256": score["predictions_sha256"],
            "score_sha256": sha256_file(args.score),
            "audit_script_sha256": sha256_file(__file__),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
