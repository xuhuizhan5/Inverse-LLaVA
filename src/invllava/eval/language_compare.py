from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from invllava.analysis.statistics import paired_bootstrap
from invllava.artifacts.hashing import sha256_file


def _load_samples(
    path: str | Path, *, metric: str, filter_name: str
) -> dict[str, tuple[float, tuple[str, str]]]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    grouped = payload.get("samples")
    if not isinstance(grouped, dict) or not grouped:
        raise ValueError(f"language result has no logged samples: {source}")

    samples: dict[str, tuple[float, tuple[str, str]]] = {}
    for task_name in sorted(grouped):
        rows = grouped[task_name]
        if not isinstance(rows, list):
            raise ValueError(f"language samples for {task_name} must be a list")
        for row in rows:
            if row.get("filter") != filter_name:
                continue
            if metric not in row:
                raise ValueError(
                    f"sample for {task_name} and filter {filter_name} has no metric {metric}"
                )
            doc_hash = str(row.get("doc_hash", ""))
            prompt_hash = str(row.get("prompt_hash", ""))
            target_hash = str(row.get("target_hash", ""))
            if not doc_hash or not prompt_hash or not target_hash:
                raise ValueError(
                    "paired language samples require document, prompt, and target hashes"
                )
            sample_id = f"{task_name}:{doc_hash}:{filter_name}"
            if sample_id in samples:
                raise ValueError(f"duplicate paired language sample: {sample_id}")
            value = row[metric]
            if not isinstance(value, (int, float)):
                raise ValueError(f"language sample metric must be numeric: {sample_id}")
            samples[sample_id] = (float(value), (prompt_hash, target_hash))
    if not samples:
        raise ValueError(f"no samples match metric={metric}, filter={filter_name}: {source}")
    return samples


def compare_language_results(
    left_path: str | Path,
    right_path: str | Path,
    *,
    metric: str,
    filter_name: str,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    left = _load_samples(left_path, metric=metric, filter_name=filter_name)
    right = _load_samples(right_path, metric=metric, filter_name=filter_name)
    if left.keys() != right.keys():
        raise ValueError("paired language results have different sample identities")
    ids = sorted(left)
    mismatched = [sample_id for sample_id in ids if left[sample_id][1] != right[sample_id][1]]
    if mismatched:
        raise ValueError(
            "paired language results have different prompts or targets; "
            f"first mismatch: {mismatched[0]}"
        )

    left_values = np.asarray([left[sample_id][0] for sample_id in ids], dtype=float)
    right_values = np.asarray([right[sample_id][0] for sample_id in ids], dtype=float)
    interval = paired_bootstrap(
        left_values,
        right_values,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    return {
        "left_result": str(Path(left_path)),
        "left_sha256": sha256_file(left_path),
        "right_result": str(Path(right_path)),
        "right_sha256": sha256_file(right_path),
        "metric": metric,
        "filter": filter_name,
        "sample_count": len(ids),
        "left_mean": float(left_values.mean()),
        "right_mean": float(right_values.mean()),
        **asdict(interval),
    }
