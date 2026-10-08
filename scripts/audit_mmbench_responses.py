#!/usr/bin/env python3
"""Separate extraction failures and rotation errors in accepted MMBench answers.

The declared scorer is unchanged. The parsing ceiling is a diagnostic bound:
it assumes every unparsed answer could become correct, without changing any
already parsed answer. It is never an alternative benchmark result.
The case-only diagnostic uppercases unparsed single-letter option labels;
it uses no reference answers to decide which responses to transform.
"""

from __future__ import annotations

import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.mmbench import MMBenchCircularProtocol, infer_mmbench_choice
from invllava.eval.records import PredictionStore
from invllava.eval.types import EvaluationExample
from invllava.eval.validation import validate_coverage, validate_frozen_prompts


def audit_responses(examples: list[EvaluationExample], predictions: dict[str, str]) -> dict:
    ids = [example.id for example in examples]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("unique, complete example/prediction coverage is required")
    score = MMBenchCircularProtocol("mmbench-response-audit").score(predictions, examples)
    groups = defaultdict(list)
    answers = Counter()
    invalid_ids = []
    case_predictions = dict(predictions)
    case_changed_ids = []
    rotation_correct = base_correct = 0
    for example in examples:
        answer = infer_mmbench_choice(predictions[example.id], example.choices)
        hit = answer == example.references[0].strip().upper()
        answers[answer or "unparsed"] += 1
        if answer is None:
            invalid_ids.append(example.id)
            candidate = predictions[example.id].strip()
            if len(candidate) == 1 and candidate in string.ascii_lowercase[: len(example.choices)]:
                case_predictions[example.id] = candidate.upper()
                case_changed_ids.append(example.id)
        rotation_correct += hit
        base_correct += hit and example.id == example.group_id
        groups[example.group_id].append((answer, hit))
    patterns = Counter()
    for rows in groups.values():
        if all(hit for _, hit in rows):
            patterns["all_correct"] += 1
        elif all(answer is None or hit for answer, hit in rows):
            patterns["unparsed_only"] += 1
        else:
            patterns["has_parsed_error"] += 1
    case_score = MMBenchCircularProtocol("mmbench-case-diagnostic").score(
        case_predictions, examples
    )
    recovered_groups = [
        group_id
        for group_id, hit in case_score.details["per_item"].items()
        if hit and not score.details["per_item"][group_id]
    ]
    return {
        "circular_accuracy": score.value,
        "circular_groups": score.count,
        "rotations": len(examples),
        "base_rotation_accuracy": base_correct / score.count,
        "all_rotation_accuracy": rotation_correct / len(examples),
        "extracted_answers": dict(sorted(answers.items())),
        "unparsed_ids": invalid_ids,
        "group_outcomes": dict(sorted(patterns.items())),
        "parsing_only_max_gain_pp": 100 * patterns["unparsed_only"] / score.count,
        "category_circular_accuracy": score.details["category_accuracy"],
        "case_only_diagnostic": {
            "changed_ids": case_changed_ids,
            "changed_rotations": len(case_changed_ids),
            "recovered_group_ids": recovered_groups,
            "recovered_groups": len(recovered_groups),
            "circular_accuracy": case_score.value,
            "gain_pp": 100 * (case_score.value - score.value),
            "remaining_unparsed": case_score.details["invalid_rotations"],
            "category_circular_accuracy": case_score.details["category_accuracy"],
            "scope": (
                "Posthoc format diagnostic, not the declared benchmark result. Uppercase only "
                "unparsed one-character valid option labels; leave parsed answers unchanged. "
                "Selection uses response and options only, never references."
            ),
        },
        "scope": "Descriptive audit; the parsing ceiling assumes all unparsed answers correct.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    score = json.loads(args.score.read_text())
    if score.get("benchmark") not in {"mmbench-en", "mmbench-cn"}:
        raise ValueError("an MMBench score is required")
    for key, path in (("examples_sha256", args.examples), ("predictions_sha256", args.predictions)):
        if score[key] != sha256_file(path):
            raise ValueError(f"score input changed: {key}")
    examples = load_examples(args.examples)
    records = list(
        PredictionStore(
            args.predictions, protocol_id=score["protocol_id"], checkpoint_id=score["checkpoint_id"]
        )
    )
    validate_coverage(records, {example.id for example in examples})
    validate_frozen_prompts(records, examples)
    predictions = {row.sample_id: row.prediction for row in records}
    rescored = MMBenchCircularProtocol(score["scorer_id"]).score(predictions, examples)
    if (rescored.value, rescored.count, rescored.details) != (
        score["value"],
        score["count"],
        score["details"],
    ):
        raise ValueError("saved score differs from the declared scorer")
    report = audit_responses(examples, predictions)
    report.update(
        status="passed",
        benchmark=score["benchmark"],
        checkpoint_id=score["checkpoint_id"],
        score_sha256=sha256_file(args.score),
        predictions_sha256=sha256_file(args.predictions),
        examples_sha256=sha256_file(args.examples),
        script_sha256=sha256_file(__file__),
    )
    atomic_write_json(args.output, report)
    printable = {key: value for key, value in report.items() if key != "unparsed_ids"}
    printable["case_only_diagnostic"] = {
        key: value
        for key, value in report["case_only_diagnostic"].items()
        if key not in {"changed_ids", "recovered_group_ids"}
    }
    print(json.dumps(printable))


if __name__ == "__main__":
    main()
