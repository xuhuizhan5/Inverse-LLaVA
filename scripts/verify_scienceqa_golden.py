#!/usr/bin/env python3
"""Bind prepared ScienceQA inputs to LLaVA's released question artifact.

This check uses the released conversations directly. A directory name or a
protocol string in an earlier manifest cannot substitute for matching inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt

SINGLE_PRED_SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def compare_official_inputs(
    questions: list[dict], examples: list[EvaluationExample]
) -> dict[str, object]:
    official = {str(row["id"]): row for row in questions if "image" in row}
    prepared = {row.id: row for row in examples}
    if len(official) != sum("image" in row for row in questions):
        raise ValueError("duplicate official ScienceQA image-question IDs")
    if len(prepared) != len(examples):
        raise ValueError("duplicate prepared ScienceQA IDs")
    missing = sorted(official.keys() - prepared.keys())
    extra = sorted(prepared.keys() - official.keys())
    mismatches = []
    for identifier in sorted(official.keys() & prepared.keys()):
        conversations = official[identifier]["conversations"]
        if len(conversations) != 2 or [row["from"] for row in conversations] != ["human", "gpt"]:
            raise ValueError("unexpected official ScienceQA conversation")
        example = prepared[identifier]
        expected = format_vicuna_v1_user_prompt(conversations[0]["value"] + SINGLE_PRED_SUFFIX)
        fields = []
        if example.prompt != expected:
            fields.append("prompt")
        if example.references != (conversations[1]["value"],):
            fields.append("reference")
        if len(example.images) != 1 or not example.choices:
            fields.append("image_or_choices")
        if fields:
            mismatches.append({"sample_id": identifier, "fields": fields})
    return {
        "passed": bool(official) and not missing and not extra and not mismatches,
        "official_image_questions": len(official),
        "prepared_questions": len(prepared),
        "missing_ids": missing,
        "extra_ids": extra,
        "mismatches": mismatches,
        "scope": "image-question IDs, released user prompts, answer references, and image flags",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-questions", type=Path, required=True)
    parser.add_argument("--official-questions-sha256", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    digest = sha256_file(args.official_questions)
    if digest != args.official_questions_sha256:
        raise ValueError("official question artifact checksum mismatch")
    result = compare_official_inputs(
        json.loads(args.official_questions.read_text()), load_examples(args.examples)
    )
    result.update(
        {
            "official_questions_sha256": digest,
            "examples_sha256": sha256_file(args.examples),
            "official_runner": "https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/sqa.sh",
        }
    )
    atomic_write_json(args.output, result)
    print(
        {
            key: value
            for key, value in result.items()
            if key not in {"missing_ids", "extra_ids", "mismatches"}
        }
    )
    if not result["passed"]:
        raise ValueError(f"ScienceQA inputs differ from the official artifact; see {args.output}")


if __name__ == "__main__":
    main()
