#!/usr/bin/env python3
"""Add row-sampling uncertainty to checksum-verified representation captures."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from invllava.analysis.representations import cka_sample_uncertainty
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256
from invllava.eval.datasets import load_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", nargs=2, action="append", required=True, metavar=("NAME", "NPZ")
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    examples = load_examples(args.examples)
    ids = [sample.id for sample in examples]
    if any(len(sample.images) != 1 for sample in examples):
        raise ValueError("this panel requires one image per example")
    image_hashes = [sha256_file(sample.images[0]) for sample in examples]
    if len(set(ids)) != len(ids) or len(set(image_hashes)) != len(ids):
        raise ValueError(
            "row bootstrap requires unique IDs and images; use grouped resampling otherwise"
        )
    captures, provenance = {}, {}
    condition = None
    for name, filename in args.capture:
        if name in captures:
            raise ValueError("capture names must be distinct")
        path = Path(filename)
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        if metadata["sample_ids"] != ids or metadata["examples_sha256"] != sha256_file(
            args.examples
        ):
            raise ValueError("capture rows/input identity differ")
        if metadata["artifact_sha256"] != sha256_file(path):
            raise ValueError("capture checksum differs")
        if metadata["batch_size"] != 1 or not metadata.get("input_condition"):
            raise ValueError("this panel requires batch-one captures with a declared condition")
        if condition is None:
            condition = metadata["input_condition"]
        elif metadata["input_condition"] != condition:
            raise ValueError("capture input conditions differ")
        with np.load(path, allow_pickle=False) as arrays:
            if arrays["sample_ids"].astype(str).tolist() != ids:
                raise ValueError("capture tensor rows differ from declared sample IDs")
            captures[name] = {layer: arrays[f"hidden.last.{layer}"] for layer in args.layers}
        provenance[name] = {
            "path": str(path),
            "metadata_sha256": sha256_file(metadata_path),
            **metadata,
        }
    if len(captures) < 2:
        raise ValueError("at least two captures are required")
    results = []
    for left, right in itertools.combinations(captures, 2):
        for layer in args.layers:
            result = cka_sample_uncertainty(
                captures[left][layer],
                captures[right][layer],
                resamples=args.resamples,
                seed=args.seed,
            )
            results.append({"left": left, "right": right, "layer": layer, **result})
            print(left, right, layer, result, flush=True)
    # Holm adjustment for the declared family of pair/layer permutation tests.
    ordered = sorted(range(len(results)), key=lambda i: results[i]["one_sided_p_value"])
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(
            running, min(1.0, (len(results) - rank) * results[index]["one_sided_p_value"])
        )
        results[index]["holm_adjusted_p_value"] = running
    atomic_write_json(
        args.output,
        {
            "execution_source_sha256": execution_source_sha256(Path(__file__).resolve().parents[1])[
                0
            ],
            "examples_sha256": sha256_file(args.examples),
            "pooling": "last valid prompt token; one-based decoder output indices",
            "layers": args.layers,
            "input_condition": condition,
            "captures": provenance,
            "image_hashes": image_hashes,
            "results": results,
            "scope": (
                "Conditional on these fitted models and sampled images; similarity does not "
                "establish grounding or superiority. Percentile intervals are pointwise; "
                "high-dimensional and duplicate-row bootstrap bias remains possible."
            ),
        },
    )


if __name__ == "__main__":
    main()
