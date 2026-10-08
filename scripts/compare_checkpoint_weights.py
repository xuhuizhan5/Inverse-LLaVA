#!/usr/bin/env python3
"""Measure stored-weight changes between matching safetensors checkpoints on CPU.

This inspects saved parameter coordinates, without loading a backbone or merging
LoRA. Coordinate norms are not activation distances or effective LoRA operator
norms. Unchanged stored values can coexist with changes in FP32 optimizer state.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file


def tensor_statistics(
    reference: torch.Tensor, candidate: torch.Tensor, *, chunk_size: int = 262144
) -> dict:
    """Use bounded FP64 reductions; preserve the original storage dtype in output."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError("tensor shape or storage dtype differs")
    if reference.device.type != "cpu" or candidate.device.type != "cpu":
        raise ValueError("checkpoint inspection must run on CPU")
    if not reference.is_floating_point() or chunk_size < 1:
        raise ValueError("floating tensors and a positive chunk size are required")
    count, changed, reference_squared, change_squared, maximum = reference.numel(), 0, 0.0, 0.0, 0.0
    left, right = reference.detach().reshape(-1), candidate.detach().reshape(-1)
    for start in range(0, count, chunk_size):
        a = left[start : start + chunk_size].double()
        b = right[start : start + chunk_size].double()
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("checkpoint contains nonfinite values")
        delta = b - a
        changed += int(torch.count_nonzero(delta))
        reference_squared += float(torch.sum(a.square()))
        change_squared += float(torch.sum(delta.square()))
        maximum = max(maximum, float(delta.abs().max()))
    if not all(math.isfinite(value) for value in (reference_squared, change_squared, maximum)):
        raise ValueError("nonfinite weight-change reduction")
    result = {
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "elements": count,
        "changed_elements": changed,
        "reference_squared_norm": reference_squared,
        "change_squared_norm": change_squared,
        "maximum_absolute_change": maximum,
    }
    if count <= 8:
        result.update(reference_values=left.tolist(), candidate_values=right.tolist())
    return result


def group_name(name: str) -> str:
    for adapter in ("lora_a", "lora_b"):
        if f".{adapter}." in name:
            return adapter
    if ".fusion." in name:
        return "fusion." + name.split(".fusion.", 1)[1].split(".", 1)[0]
    return "other"


def compare_files(reference: Path, candidate: Path) -> dict:
    hashes = {"reference": sha256_file(reference), "candidate": sha256_file(candidate)}
    records = {}
    totals = defaultdict(
        lambda: {
            "elements": 0,
            "changed_elements": 0,
            "reference_squared_norm": 0.0,
            "change_squared_norm": 0.0,
        }
    )
    with (
        safe_open(reference, framework="pt", device="cpu") as left,
        safe_open(candidate, framework="pt", device="cpu") as right,
    ):
        if not left.keys() or set(left.keys()) != set(right.keys()):
            raise ValueError("checkpoint parameter names differ or are empty")
        for name in sorted(left.keys()):
            row = tensor_statistics(left.get_tensor(name), right.get_tensor(name))
            records[name] = row
            for key in totals[group_name(name)]:
                totals[group_name(name)][key] += row[key]
    groups = {}
    for name, row in sorted(totals.items()):
        denominator = row["reference_squared_norm"]
        groups[name] = {
            **row,
            "changed_fraction": row["changed_elements"] / row["elements"]
            if row["elements"]
            else None,
            "relative_l2_change": math.sqrt(row["change_squared_norm"] / denominator)
            if denominator
            else None,
        }
    if hashes != {"reference": sha256_file(reference), "candidate": sha256_file(candidate)}:
        raise RuntimeError("checkpoint changed during inspection")
    return {
        "status": "passed",
        "reference": str(reference),
        "candidate": str(candidate),
        "checkpoint_sha256": hashes,
        "tensor_count": len(records),
        "groups": groups,
        "tensors": records,
        "scope": "CPU stored-coordinate comparison. Numerically changed elements, FP64 reductions. "
        "Zero-reference relative norms are undefined. LoRA factors are separate; "
        "their changes are not merged operator norms. "
        "No optimizer-state, causal, or functional-equivalence inference.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    result = compare_files(args.reference, args.candidate)
    atomic_write_json(args.output, {**result, "script_sha256": sha256_file(__file__)})
    print(args.output)


if __name__ == "__main__":
    main()
