#!/usr/bin/env python3
"""Describe answer bias and paired-question correctness in a completed MME run.

This reuses the benchmark parser and scoring formula. Constant-answer controls
describe the metric's floor; they are not substitutes for image interventions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.mme import MMEProtocol, extract_mme_answer
from invllava.eval.types import EvaluationExample


def audit_responses(examples: list[EvaluationExample], predictions: dict[str, str]) -> dict:
    ids = [example.id for example in examples]
    if not ids or len(set(ids)) != len(ids) or set(predictions) != set(ids):
        raise ValueError("unique, complete example/prediction coverage is required")
    domains = {example.metadata.get("domain") for example in examples}
    if len(domains) != 1 or next(iter(domains)) not in {"cognition", "perception"}:
        raise ValueError("exactly one declared MME domain is required")
    scorer = MMEProtocol("mme-response-audit", domain=next(iter(domains)))
    score = scorer.score(predictions, examples)
    categories = defaultdict(list)
    for example in examples:
        categories[example.metadata["category"]].append(example)
    results = {}
    for category, members in sorted(categories.items()):
        pairs = defaultdict(list)
        confusion = Counter()
        answers = Counter()
        for example in members:
            answer = extract_mme_answer(predictions[example.id]) or "invalid"
            reference = example.references[0].strip().lower()
            answers[answer] += 1
            confusion[f"reference_{reference}/prediction_{answer}"] += 1
            pairs[example.group_id or example.id].append(answer == reference)
        results[category] = {
            "questions": len(members),
            "answers": dict(answers),
            "confusion": dict(confusion),
            "pairs": len(pairs),
            "both_correct_pairs": sum(all(pair) for pair in pairs.values()),
            "score": score.details["category_scores"][category],
        }
    return {
        "domain": next(iter(domains)),
        "score": score.value,
        "categories": results,
        "constant_answer_controls": {
            answer: scorer.score(dict.fromkeys(ids, answer), examples).value
            for answer in ("yes", "no")
        },
        "scope": "Descriptive answer-pattern audit; image dependence requires an intervention.",
    }


def category_comparisons(
    examples: list[EvaluationExample],
    left: dict,
    right: dict,
    *,
    resamples: int = 10000,
    seed: int = 2026,
) -> dict:
    """Report every category using MME image-pair resampling and its native score."""
    import numpy as np

    from invllava.analysis.statistics import mme_paired_bootstrap

    for field in ("benchmark", "examples_sha256", "protocol_id", "scorer_id"):
        if not left.get(field) or left[field] != right.get(field):
            raise ValueError(f"MME comparison has incompatible {field}")
    if left["benchmark"] not in {"mme-cognition", "mme-perception"}:
        raise ValueError("MME score artifacts are required")
    ids = [example.id for example in examples]
    a, b = left["details"]["per_item"], right["details"]["per_item"]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(a) or set(ids) != set(b):
        raise ValueError("category comparison requires unique complete example coverage")
    if any(value not in (0, 1) for items in (a, b) for value in items.values()):
        raise ValueError("MME per-question correctness must be binary")
    categories = defaultdict(list)
    for example in examples:
        category = example.metadata.get("category")
        if not category or example.group_id is None:
            raise ValueError("MME category and image-pair identity are required")
        categories[category].append(example)
    result = {}
    for category, members in sorted(categories.items()):
        groups = np.asarray([example.group_id for example in members])
        interval = mme_paired_bootstrap(
            np.asarray([a[example.id] for example in members]),
            np.asarray([b[example.id] for example in members]),
            groups,
            np.asarray([category] * len(members)),
            resamples=resamples,
            seed=seed,
        )
        result[category] = {
            **asdict(interval),
            "questions": len(members),
            "image_groups": len(set(groups.tolist())),
        }
    if not math.isclose(
        sum(row["estimate"] for row in result.values()),
        left["value"] - right["value"],
        abs_tol=1e-8,
    ):
        raise ValueError("category differences do not reproduce the aggregate score difference")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument(
        "--reference-score",
        type=Path,
        help="Compare every category with a compatible completed score",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = json.loads(args.score.read_text())
    if score["examples_sha256"] != sha256_file(args.examples):
        raise ValueError("score does not bind the supplied examples")
    if score["predictions_sha256"] != sha256_file(args.predictions):
        raise ValueError("score does not bind the supplied predictions")
    records = [json.loads(line) for line in args.predictions.read_text().splitlines() if line]
    predictions = {record["sample_id"]: record["prediction"] for record in records}
    if len(predictions) != len(records):
        raise ValueError("duplicate prediction IDs")
    if any(
        record["checkpoint_id"] != score["checkpoint_id"]
        or record["protocol_id"] != score["protocol_id"]
        for record in records
    ):
        raise ValueError("mixed prediction checkpoint/protocol identity")
    examples = load_examples(args.examples)
    result = audit_responses(examples, predictions)
    if abs(result["score"] - score["value"]) > 1e-9:
        raise ValueError("recomputed score differs from the retained score")
    if args.reference_score:
        reference = json.loads(args.reference_score.read_text())
        result["category_comparisons"] = category_comparisons(
            examples,
            score,
            reference,
            resamples=args.resamples,
            seed=args.seed,
        )
        result["reference_score_sha256"] = sha256_file(args.reference_score)
        result["comparison_scope"] = (
            "All categories, candidate minus reference; native MME score points. "
            "Conditional checkpoint intervals, without multiplicity correction or seed variation."
        )
    atomic_write_json(
        args.output,
        {
            **result,
            "checkpoint_id": score["checkpoint_id"],
            "protocol_id": score["protocol_id"],
            "examples_sha256": score["examples_sha256"],
            "predictions_sha256": score["predictions_sha256"],
            "score_sha256": sha256_file(args.score),
            "audit_script_sha256": sha256_file(__file__),
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
