#!/usr/bin/env python3
"""Rescore immutable MME answers with the benchmark authors' calculator.

Run on an execution host. The pinned archive is supplied explicitly; this
script downloads nothing and never overwrites a prediction or score artifact.
It executes the reviewed calculator in a temporary directory, including its
line-based pairing and answer parser, independently of lmms-eval.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import re
import tempfile
import types
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.protocols.mme import MMEProtocol, extract_mme_answer
from invllava.eval.types import EvaluationExample

OFFICIAL_TOOL_SHA256 = "b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee"
OFFICIAL_TOOL_URL = (
    "https://raw.githubusercontent.com/BradyFU/"
    "Awesome-Multimodal-Large-Language-Models/Evaluation/tools/eval_tool.zip"
)


def grouped_rows(paths: list[Path]) -> dict[str, list[dict]]:
    """Validate identity and arrange each image's two questions adjacently."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: set[str] = set()
    checkpoints: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"duplicate sample: {sample_id}")
            seen.add(sample_id)
            checkpoints.add(row["checkpoint_id"])
            category = row["metadata"]["category"]
            if not sample_id.startswith(category + "/"):
                raise ValueError(f"category mismatch: {sample_id}")
            if len(row["references"]) != 1 or row["references"][0].lower() not in {"yes", "no"}:
                raise ValueError(f"nonbinary reference: {sample_id}")
            # The original converter writes verbatim answers to four-column TSV.
            # A multiline/tabbed response requires a separate protocol decision.
            if any(value in row["prediction"] for value in ("\t", "\n", "\r")):
                raise ValueError(f"answer cannot enter official TSV verbatim: {sample_id}")
            groups[category, sample_id.rsplit("/", 1)[0]].append(row)
    if len(checkpoints) != 1:
        raise ValueError("one checkpoint per comparison is required")
    categories: dict[str, list[dict]] = defaultdict(list)
    for (category, group), rows in sorted(groups.items()):
        if {row["sample_id"].rsplit("/", 1)[1] for row in rows} != {"0", "1"}:
            raise ValueError(f"incomplete image pair: {group}")
        if len(rows) != 2:
            raise ValueError(f"expected two questions: {group}")
        categories[category].extend(sorted(rows, key=lambda row: row["sample_id"]))
    return dict(categories)


def audit(calculator: types.ModuleType, paths: list[Path]) -> dict:
    categories = grouped_rows(paths)
    expected_categories = {c for names in calculator.eval_type_dict.values() for c in names}
    if set(categories) != expected_categories:
        raise ValueError("both complete MME domains are required")
    metrics = calculator.calculate_metrics()
    parser_differences = []
    local_categories = {}
    for domain, names in calculator.eval_type_dict.items():
        rows = [row for name in names for row in categories[name]]
        expected_count = 2114 if domain == "Perception" else 260
        if len(rows) != expected_count:
            raise ValueError(f"incomplete {domain} coverage: {len(rows)}")
        examples = [
            EvaluationExample(
                id=row["sample_id"],
                prompt=row["prompt"],
                images=(),
                references=tuple(row["references"]),
                group_id=row["sample_id"].rsplit("/", 1)[0],
                metadata={"category": row["metadata"]["category"], "domain": domain.lower()},
            )
            for row in rows
        ]
        local = MMEProtocol("audit", domain=domain.lower()).score(
            {row["sample_id"]: row["prediction"] for row in rows},
            examples,
        )
        local_categories.update(local.details["category_scores"])
        for row in rows:
            # process_result lowercases TSV fields and retains the final newline.
            official = metrics.parse_pred_ans(row["prediction"].lower() + "\n")
            local_answer = extract_mme_answer(row["prediction"]) or "other"
            if official != local_answer:
                parser_differences.append(
                    {
                        "sample_id": row["sample_id"],
                        "prediction": row["prediction"],
                        "official": official,
                        "repository": local_answer,
                    }
                )
    with tempfile.TemporaryDirectory(prefix="invllava-mme-official-") as directory:
        root = Path(directory)
        for category, rows in categories.items():
            # Question contents are irrelevant to this scorer; IDs and reference
            # bindings have separate complete released-fixture golden checks.
            (root / f"{category}.txt").write_text(
                "".join(
                    f"{row['sample_id'].rsplit('/', 1)[0]}\t{row['sample_id']}\t"
                    f"{row['references'][0]}\t{row['prediction']}\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics.process_result(str(root))
    output = captured.getvalue()
    official_categories = {
        category: float(score)
        for category, score in re.findall(
            r"^\s*(\w+)\s+score:\s+([\d.eE+-]+)\s*$",
            output,
            flags=re.MULTILINE,
        )
        if category in expected_categories
    }
    if set(official_categories) != expected_categories:
        raise ValueError("could not recover all official calculator category scores")
    deltas = {c: local_categories[c] - official_categories[c] for c in expected_categories}
    return {
        "checkpoint_id": next(iter(categories.values()))[0]["checkpoint_id"],
        "prediction_files": [{"path": str(p), "sha256": sha256_file(p)} for p in paths],
        "questions": sum(map(len, categories.values())),
        "parser_differences": parser_differences,
        "repository_minus_official": deltas,
        "score_match": all(math.isclose(d, 0, abs_tol=1e-10) for d in deltas.values()),
        "official_categories": official_categories,
        "official_totals": {
            domain: sum(official_categories[c] for c in names)
            for domain, names in calculator.eval_type_dict.items()
        },
        "calculator_stdout": output,
        "calculator_warnings": [
            {"category": warning.category.__name__, "message": str(warning.message)}
            for warning in caught
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-tool", required=True, type=Path)
    parser.add_argument(
        "--run",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "PERCEPTION_JSONL", "COGNITION_JSONL"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.official_tool) != OFFICIAL_TOOL_SHA256:
        raise ValueError("official archive hash differs from the reviewed calculator")
    with zipfile.ZipFile(args.official_tool) as archive:
        source = archive.read("eval_tool/calculation.py").decode("utf-8")
    calculator = types.ModuleType("official_mme_calculation")
    exec(compile(source, "official_mme/calculation.py", "exec"), calculator.__dict__)
    results = {}
    for label, perception, cognition in args.run:
        if label in results:
            raise ValueError(f"duplicate run label: {label}")
        results[label] = audit(calculator, [Path(perception), Path(cognition)])
    report = {
        "official_tool_url": OFFICIAL_TOOL_URL,
        "official_tool_sha256": OFFICIAL_TOOL_SHA256,
        "scope": "Original MME calculator on saved answers; inference not repeated.",
        "runs": results,
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                label: {
                    "score_match": r["score_match"],
                    "parser_differences": len(r["parser_differences"]),
                    "official_totals": r["official_totals"],
                }
                for label, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
