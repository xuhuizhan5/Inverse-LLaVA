"""Compare native parsing and retained scores with the pinned official ScienceQA code."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.protocols.scienceqa import extract_llava_scienceqa_choice

OFFICIAL_SHA256 = "e57b92cafc7a4c0d0228bc0554a4315ef808e844d009dcaafa3b0f6ac693fbaf"
OFFICIAL_REVISION = "c121f0432da27facab705978f83c4ada465e46fd"


def official_parser(path: Path):
    """Execute only the extraction block and index helper from verified source."""
    if sha256_file(path) != OFFICIAL_SHA256:
        raise ValueError("official ScienceQA source identity changed")
    tree = ast.parse(path.read_text())
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_pred_idx"
    ]
    blocks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "pred_text"
    ]
    if len(helpers) != 1 or len(blocks) != 1:
        raise ValueError("unexpected official scorer structure")
    helper_code = compile(ast.Module(body=helpers, type_ignores=[]), str(path), "exec")
    parse_code = compile(ast.Module(body=blocks, type_ignores=[]), str(path), "exec")
    helpers_namespace = {}
    exec(helper_code, helpers_namespace)
    options = list("ABCDE")

    def extract(value, count):
        context = {"pred_text": value, "args": SimpleNamespace(options=options), "re": re}
        exec(parse_code, context)
        index = helpers_namespace["get_pred_idx"](context["answer"], range(count), options)
        return None if index < 0 else options[index]

    return extract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-scorer", type=Path, required=True)
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
    reference = official_parser(args.official_scorer)
    examples = load_examples(args.examples)
    by_id = {row.id: row for row in examples}
    if not examples or len(by_id) != len(examples):
        raise ValueError("examples must have unique IDs and nonempty coverage")
    results = {}
    for label, predictions_path, score_path in args.series:
        if label in results:
            raise ValueError("duplicate series label")
        predictions, score_file = Path(predictions_path), Path(score_path)
        score = json.loads(score_file.read_text())
        if sha256_file(predictions) != score["predictions_sha256"]:
            raise ValueError("prediction checksum differs from score")
        if sha256_file(args.examples) != score["examples_sha256"]:
            raise ValueError("example checksum differs from score")
        rows = [json.loads(line) for line in predictions.read_text().splitlines()]
        if len(rows) != len(by_id) or {row["sample_id"] for row in rows} != set(by_id):
            raise ValueError("predictions must cover the exact unique example IDs")
        differences, correct, invalid = [], 0, 0
        for row in rows:
            example = by_id[row["sample_id"]]
            if row["prompt"] != example.prompt or tuple(row["references"]) != example.references:
                raise ValueError("prediction inputs differ from examples")
            choice = reference(row["prediction"], len(example.choices))
            native = extract_llava_scienceqa_choice(row["prediction"], len(example.choices))
            value = int(choice == example.references[0])
            correct += value
            invalid += choice is None
            if choice != native or value != score["details"]["per_item"][example.id]:
                differences.append(
                    {
                        "sample_id": example.id,
                        "response": row["prediction"],
                        "official_choice": choice,
                        "native_choice": native,
                        "official_score": value,
                        "stored_score": score["details"]["per_item"][example.id],
                    }
                )
        passed = (
            not differences
            and score["count"] == len(rows)
            and invalid == score["details"]["invalid"]
            and math.isclose(correct / len(rows), score["value"], rel_tol=0, abs_tol=1e-12)
        )
        results[label] = {
            "passed": passed,
            "count": len(rows),
            "official_correct": correct,
            "official_invalid": invalid,
            "mismatch_count": len(differences),
            "mismatch_examples": differences[:100],
            "predictions_sha256": sha256_file(predictions),
            "score_sha256": sha256_file(score_file),
        }
    payload = {
        "status": "passed" if all(row["passed"] for row in results.values()) else "failed",
        "official_source_sha256": OFFICIAL_SHA256,
        "official_revision": OFFICIAL_REVISION,
        "verifier_sha256": sha256_file(__file__),
        "examples_sha256": sha256_file(args.examples),
        "series": results,
        "scope": (
            "Exact pinned official extraction and retained per-item/aggregate scores; "
            "raw responses unchanged."
        ),
    }
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "series": len(results),
                "items": sum(row["count"] for row in results.values()),
                "mismatches": sum(row["mismatch_count"] for row in results.values()),
            }
        )
    )
    if payload["status"] != "passed":
        raise ValueError(f"ScienceQA scorer mismatch; retained diagnostics at {args.output}")


if __name__ == "__main__":
    main()
