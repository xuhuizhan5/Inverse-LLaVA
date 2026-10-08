#!/usr/bin/env python3
"""Rescore published answers against separately obtained official annotations.

Published answers omit dataset text and labels. Per-prompt hashes verify that
the locally prepared examples use the exact input protocol before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.identifiers import benchmark_protocol_id
from invllava.config.schema import BenchmarkSpec
from invllava.eval.datasets import load_examples
from invllava.eval.protocols import build_protocol
from invllava.eval.types import EvaluationExample


def validated_answers(
    records: list[dict], examples: list[EvaluationExample], protocol_id: str
) -> dict[str, str]:
    expected = {example.id: example for example in examples}
    if len(expected) != len(examples) or not expected:
        raise ValueError("examples must have nonempty, unique sample IDs")
    if len({record["checkpoint_id"] for record in records}) != 1:
        raise ValueError("answers must describe exactly one checkpoint")
    answers = {}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id not in expected or sample_id in answers:
            raise ValueError(f"unexpected or duplicate sample: {sample_id}")
        if record["scoring_protocol_id"] != protocol_id:
            raise ValueError("answer protocol does not match the benchmark configuration")
        prompt_hash = hashlib.sha256(expected[sample_id].prompt.encode()).hexdigest()
        if record["prompt_sha256"] != prompt_hash:
            raise ValueError(f"prompt mismatch for {sample_id}")
        if not isinstance(record["prediction"], str):
            raise ValueError(f"prediction must be text for {sample_id}")
        answers[sample_id] = record["prediction"]
    if answers.keys() != expected.keys():
        raise ValueError("answer coverage differs from the prepared examples")
    return answers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark")
    parser.add_argument("--answers", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    spec = BenchmarkSpec.model_validate(yaml.safe_load(Path(args.benchmark).read_text()))
    if spec.external_only or spec.verification_status != "golden_verified":
        raise ValueError(
            "use a golden-verified local benchmark; server scores need official scoring"
        )
    protocol = build_protocol(spec)
    records = [json.loads(line) for line in Path(args.answers).read_text().splitlines() if line]
    examples = load_examples(args.examples)
    protocol_id = benchmark_protocol_id(spec)
    answers = validated_answers(records, examples, protocol_id)
    score = protocol.score(answers, examples)
    result = {
        "benchmark": spec.id,
        "protocol_id": protocol_id,
        "checkpoint_id": records[0]["checkpoint_id"],
        "scorer_id": protocol.id,
        "answers_sha256": sha256_file(args.answers),
        "examples_sha256": sha256_file(args.examples),
        "protocol_config_sha256": sha256_file(args.benchmark),
        **score.__dict__,
    }
    atomic_write_json(args.output, result)
    print(json.dumps({"value": score.value, "count": score.count}))


if __name__ == "__main__":
    main()
