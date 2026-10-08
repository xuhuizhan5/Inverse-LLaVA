#!/usr/bin/env python3
"""Compare scientific prediction content while ignoring operational metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file

_DEFAULT_FIELDS = ("prompt", "prediction", "references", "metadata", "generation")


def _index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            if sample_id in result:
                raise ValueError(f"duplicate sample ID {sample_id} in {path}:{line_number}")
            result[sample_id] = row
    if not result:
        raise ValueError(f"prediction file is empty: {path}")
    return result


def compare_content(
    left_path: Path,
    right_path: Path,
    fields: tuple[str, ...] = _DEFAULT_FIELDS,
    *,
    id_policy: str = "exact",
) -> dict[str, Any]:
    """Compare exact panels, or every right-hand row against a larger left panel."""
    if not fields or len(fields) != len(set(fields)):
        raise ValueError("comparison fields must be non-empty and unique")
    if id_policy not in ("exact", "right-subset"):
        raise ValueError("unsupported sample-ID policy")

    left = _index(left_path)
    right = _index(right_path)
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    mismatches: list[dict[str, Any]] = []
    for sample_id in sorted(set(left) & set(right)):
        for field in fields:
            absent = [
                side
                for side, row in (("left", left[sample_id]), ("right", right[sample_id]))
                if field not in row
            ]
            if absent or left[sample_id][field] != right[sample_id][field]:
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "field": field,
                        "left": left[sample_id].get(field),
                        "right": right[sample_id].get(field),
                        "missing_in": absent,
                    }
                )
    return {
        "format": "invllava-prediction-content-parity-v1",
        "verifier_sha256": sha256_file(Path(__file__)),
        "left": str(left_path.resolve()),
        "left_sha256": sha256_file(left_path),
        "right": str(right_path.resolve()),
        "right_sha256": sha256_file(right_path),
        "id_policy": id_policy,
        "fields": list(fields),
        "left_count": len(left),
        "right_count": len(right),
        "compared_count": len(set(left) & set(right)),
        "missing_left_count": len(missing_left),
        "missing_right_count": len(missing_right),
        "mismatch_count": len(mismatches),
        "mismatch_examples": mismatches[:100],
        "passed": not missing_left
        and (id_policy == "right-subset" or not missing_right)
        and not mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--field", action="append", dest="fields")
    parser.add_argument(
        "--id-policy",
        choices=("exact", "right-subset"),
        default="exact",
        help="Require identical IDs, or require all right-hand IDs in the left panel.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fields = tuple(args.fields or _DEFAULT_FIELDS)
    payload = compare_content(args.left, args.right, fields, id_policy=args.id_policy)
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(
            "prediction parity failed: "
            f"missing_left={payload['missing_left_count']}, "
            f"missing_right={payload['missing_right_count']}, "
            f"mismatches={payload['mismatch_count']}"
        )
    print(f"prediction parity passed: {payload['compared_count']} samples and {len(fields)} fields")


if __name__ == "__main__":
    main()
