#!/usr/bin/env python3
"""Derive paired activation changes from a verified image-intervention capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from invllava.analysis.runtime import write_representation_artifact
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256


def derive_change(original: Path, intervened: Path, manifest_path: Path, output: Path):
    originals = json.loads(original.with_suffix(".json").read_text())
    controls = json.loads(intervened.with_suffix(".json").read_text())
    intervention = json.loads(manifest_path.read_text())
    for field in ("sample_ids", "checkpoint_id", "backend", "batch_size", "torch_version"):
        if originals[field] != controls[field]:
            raise ValueError(f"paired captures differ in {field}")
    for field in ("dtype", "attention_backend", "lora_execution", "device"):
        if originals.get(field) != controls.get(field):
            raise ValueError(f"paired capture runtimes differ in {field}")
    if intervention["mode"] not in {"blank", "shuffled"}:
        raise ValueError("unsupported image intervention")
    if (
        originals["examples_sha256"] != intervention["source_examples_sha256"]
        or controls["examples_sha256"] != intervention["examples_sha256"]
    ):
        raise ValueError("intervention does not bind these original/control inputs")
    for path, metadata in ((original, originals), (intervened, controls)):
        if sha256_file(path) != metadata["artifact_sha256"]:
            raise ValueError("capture checksum differs")
    with (
        np.load(original, allow_pickle=False) as left,
        np.load(intervened, allow_pickle=False) as right,
    ):
        ids = left["sample_ids"].astype(str).tolist()
        if ids != right["sample_ids"].astype(str).tolist() or ids != originals["sample_ids"]:
            raise ValueError("capture tensor rows are not aligned")
        keys = [key for key in left.files if key.startswith("hidden.last.")]
        if not keys or set(keys) != {key for key in right.files if key.startswith("hidden.last.")}:
            raise ValueError("decoder output keys differ or are absent")
        changes = {}
        for key in keys:
            if left[key].shape != right[key].shape or left[key].shape[0] != len(ids):
                raise ValueError("paired activation shapes differ")
            changes[key] = left[key].astype(np.float32) - right[key].astype(np.float32)
            if not np.isfinite(changes[key]).all():
                raise ValueError("activation change contains non-finite values")
    metadata = {
        key: value
        for key, value in originals.items()
        if key not in {"artifact_sha256", "arrays", "sample_ids", "sample_count", "pooling"}
    }
    metadata.update(
        {
            "input_condition": f"activation change: original-minus-{intervention['mode']}",
            "original_capture": str(original),
            "original_capture_sha256": sha256_file(original),
            "intervened_capture": str(intervened),
            "intervened_capture_sha256": sha256_file(intervened),
            "intervention_manifest_sha256": sha256_file(manifest_path),
            "derivation_source_sha256": execution_source_sha256(
                Path(__file__).resolve().parents[1]
            )[0],
            "scope": (
                "Within-model image intervention at fixed prompt. Activation differences describe "
                "sensitivity to this intervention, rather than an additive decomposition of "
                "visual information."
            ),
        }
    )
    write_representation_artifact(
        output,
        output.with_suffix(".json"),
        sample_ids=ids,
        arrays=changes,
        metadata=metadata,
        pooling="last valid prompt token: original minus intervened",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--intervened", type=Path, required=True)
    parser.add_argument("--intervention-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    derive_change(args.original, args.intervened, args.intervention_manifest, args.output)


if __name__ == "__main__":
    main()
