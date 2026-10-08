from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict
from math import isfinite
from typing import Any

import numpy as np

from invllava.analysis.statistics import grouped_paired_bootstrap, paired_bootstrap


def _number(value: Any, *, sample_id: str, metadata_key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{metadata_key} is boolean for sample {sample_id}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{metadata_key} is not numeric for sample {sample_id}") from error
    if not isfinite(number):
        raise ValueError(f"{metadata_key} is not finite for sample {sample_id}")
    return number


def _display_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _bucket_label(lower: float, upper: float | None) -> str:
    if upper is None:
        return f"{_display_number(lower)}+"
    if lower.is_integer() and upper.is_integer():
        if upper == lower + 1:
            return _display_number(lower)
        return f"{int(lower)}-{int(upper - 1)}"
    return f"[{_display_number(lower)}, {_display_number(upper)})"


def numeric_score_strata(
    scores_by_model: dict[str, dict[str, float]],
    metadata_by_id: dict[str, dict[str, Any]],
    *,
    metadata_key: str,
    boundaries: tuple[float, ...],
    primary_model: str,
    groups_by_id: dict[str, str] | None = None,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, Any]:
    """Summarize paired scores within deterministic numeric metadata buckets."""

    labels = list(scores_by_model)
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("stratified analysis requires at least two unique model labels")
    if primary_model not in scores_by_model:
        raise ValueError("primary model is absent from stratified score inputs")
    if not boundaries or any(not isfinite(value) for value in boundaries):
        raise ValueError("boundaries must be finite and non-empty")
    if any(boundaries[index + 1] <= boundaries[index] for index in range(len(boundaries) - 1)):
        raise ValueError("boundaries must be strictly increasing")

    id_sets = {frozenset(values) for values in scores_by_model.values()}
    if len(id_sets) != 1 or not next(iter(id_sets), frozenset()):
        raise ValueError("stratified score inputs must have identical non-empty sample IDs")
    sample_ids = sorted(next(iter(id_sets)))
    if set(sample_ids) != set(metadata_by_id):
        raise ValueError("metadata and score sample IDs differ")
    if any(not np.isfinite(list(scores.values())).all() for scores in scores_by_model.values()):
        raise ValueError("stratified scores must be finite")
    if groups_by_id is not None and (
        set(groups_by_id) != set(sample_ids)
        or any(not isinstance(group, str) or not group for group in groups_by_id.values())
    ):
        raise ValueError("image groups must cover every score ID with nonempty string identities")

    buckets: list[list[str]] = [[] for _ in boundaries]
    for sample_id in sample_ids:
        metadata = metadata_by_id[sample_id]
        if metadata_key not in metadata:
            raise ValueError(f"metadata key {metadata_key!r} is absent for sample {sample_id}")
        value = _number(metadata[metadata_key], sample_id=sample_id, metadata_key=metadata_key)
        index = bisect_right(boundaries, value) - 1
        if index < 0:
            raise ValueError(
                f"{metadata_key}={value:g} for sample {sample_id} is below the first boundary"
            )
        buckets[index].append(sample_id)

    strata = []
    for index, bucket_ids in enumerate(buckets):
        if not bucket_ids:
            continue
        lower = boundaries[index]
        upper = boundaries[index + 1] if index + 1 < len(boundaries) else None
        means = {
            label: float(np.mean([scores_by_model[label][sample_id] for sample_id in bucket_ids]))
            for label in labels
        }
        intervals = {}
        for label in labels:
            if label == primary_model:
                continue
            primary = np.asarray(
                [scores_by_model[primary_model][sample_id] for sample_id in bucket_ids]
            )
            reference = np.asarray([scores_by_model[label][sample_id] for sample_id in bucket_ids])
            if groups_by_id is None:
                interval = paired_bootstrap(
                    primary,
                    reference,
                    resamples=resamples,
                    confidence=confidence,
                    seed=seed,
                )
            else:
                interval = grouped_paired_bootstrap(
                    primary,
                    reference,
                    np.asarray([groups_by_id[sample_id] for sample_id in bucket_ids]),
                    resamples=resamples,
                    confidence=confidence,
                    seed=seed,
                )
            intervals[f"{primary_model}-minus-{label}"] = asdict(interval)
        strata.append(
            {
                "label": _bucket_label(lower, upper),
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "count": len(bucket_ids),
                "group_count": len({groups_by_id[item] for item in bucket_ids})
                if groups_by_id is not None
                else None,
                "mean_scores": means,
                "paired_intervals": intervals,
            }
        )
    return {
        "format": "invllava-numeric-score-strata-v1",
        "metadata_key": metadata_key,
        "boundaries": list(boundaries),
        "primary_model": primary_model,
        "sample_count": len(sample_ids),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "resampling_unit": "image_group" if groups_by_id is not None else "item",
        "strata": strata,
    }
