from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float
    confidence: float
    resamples: int
    seed: int
    unit: str


def recovery_ratio(inverse: float, recovered: float, controlled_baseline: float) -> float:
    gap = controlled_baseline - inverse
    if gap == 0:
        raise ValueError("recovery ratio is undefined when the controlled gap is zero")
    return (recovered - inverse) / gap


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Percentile CI for paired item-level differences (left minus right)."""

    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be in (0,1)")
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired inputs must be non-empty one-dimensional arrays of equal length")
    differences = left - right
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, differences.size, differences.size)
        estimates[index] = statistic(differences[sample])
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return ConfidenceInterval(
        float(statistic(differences)),
        float(low),
        float(high),
        confidence,
        resamples,
        seed,
        "item",
    )


def grouped_paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Resample whole groups while retaining the benchmark's item-weighted mean.

    Groups can contain different numbers of questions. Averaging their means
    would change the estimand to an equally weighted image score.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    groups = np.asarray(groups)
    if left.shape != right.shape or left.shape != groups.shape or left.ndim != 1:
        raise ValueError("left, right, and groups must be aligned one-dimensional vectors")
    if not left.size or resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("grouped bootstrap inputs or interval settings are invalid")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("bootstrap scores must be finite")
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    totals = np.bincount(inverse, weights=left - right)
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for index in range(resamples):
        sampled = rng.integers(0, len(counts), len(counts))
        estimates[index] = totals[sampled].sum() / counts[sampled].sum()
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return ConfidenceInterval(
        float((left - right).mean()),
        float(low),
        float(high),
        confidence,
        resamples,
        seed,
        "group",
    )


def stratified_macro_paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    strata: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Paired item bootstrap for a fixed equal-weight macro average of strata."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    strata = np.asarray(strata)
    if left.shape != right.shape or left.shape != strata.shape or left.ndim != 1:
        raise ValueError("left, right, and strata must be aligned one-dimensional vectors")
    if left.size == 0 or resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("stratified bootstrap inputs or interval settings are invalid")
    unique = np.unique(strata)
    if any(np.count_nonzero(strata == stratum) == 0 for stratum in unique):
        raise ValueError("every macro stratum must contain at least one item")

    differences = left - right
    estimate = float(np.mean([differences[strata == stratum].mean() for stratum in unique]))
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for iteration in range(resamples):
        stratum_estimates = []
        for stratum in unique:
            values = differences[strata == stratum]
            sample = rng.integers(0, len(values), len(values))
            stratum_estimates.append(float(values[sample].mean()))
        estimates[iteration] = float(np.mean(stratum_estimates))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return ConfidenceInterval(
        estimate,
        float(low),
        float(high),
        confidence,
        resamples,
        seed,
        "fixed_strata_item",
    )


def _mme_score(correct: np.ndarray, groups: np.ndarray, categories: np.ndarray) -> float:
    total = 0.0
    for category in np.unique(categories):
        category_mask = categories == category
        category_correct = correct[category_mask]
        category_groups = groups[category_mask]
        accuracy = float(category_correct.mean())
        accuracy_plus = float(
            np.mean(
                [
                    bool(np.all(category_correct[category_groups == group]))
                    for group in np.unique(category_groups)
                ]
            )
        )
        total += 100.0 * (accuracy + accuracy_plus)
    return total


def mme_paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    categories: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Paired image-group bootstrap that recomputes MME accuracy-plus."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    groups = np.asarray(groups)
    categories = np.asarray(categories)
    if not (left.shape == right.shape == groups.shape == categories.shape) or left.ndim != 1:
        raise ValueError("MME correctness, groups, and categories must be aligned vectors")
    if left.size == 0 or resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("MME bootstrap inputs or interval settings are invalid")
    estimate = _mme_score(left, groups, categories) - _mme_score(right, groups, categories)
    rng = np.random.default_rng(seed)
    estimates = np.zeros(resamples, dtype=float)
    category_statistics: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for category in np.unique(categories):
        category_mask = categories == category
        category_groups = groups[category_mask]
        category_left = left[category_mask]
        category_right = right[category_mask]
        unique_groups = np.unique(category_groups)
        counts = np.empty(len(unique_groups), dtype=np.int64)
        left_sums = np.empty(len(unique_groups), dtype=float)
        right_sums = np.empty(len(unique_groups), dtype=float)
        left_all = np.empty(len(unique_groups), dtype=float)
        right_all = np.empty(len(unique_groups), dtype=float)
        for index, group in enumerate(unique_groups):
            group_mask = category_groups == group
            left_group = category_left[group_mask]
            right_group = category_right[group_mask]
            counts[index] = np.count_nonzero(group_mask)
            left_sums[index] = left_group.sum()
            right_sums[index] = right_group.sum()
            left_all[index] = float(np.all(left_group))
            right_all[index] = float(np.all(right_group))
        category_statistics.append((counts, left_sums, right_sums, left_all, right_all))

    # Chunking bounds the largest temporary array while retaining vectorized
    # resampling. Per-group sums/counts reproduce concatenation exactly, including
    # categories whose natural groups contain different numbers of questions.
    chunk_size = 512
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        batch = stop - start
        differences = np.zeros(batch, dtype=float)
        for counts, left_sums, right_sums, left_all, right_all in category_statistics:
            group_count = len(counts)
            sampled = rng.integers(0, group_count, size=(batch, group_count))
            item_counts = counts[sampled].sum(axis=1)
            left_score = left_sums[sampled].sum(axis=1) / item_counts + left_all[sampled].mean(
                axis=1
            )
            right_score = right_sums[sampled].sum(axis=1) / item_counts + right_all[sampled].mean(
                axis=1
            )
            differences += 100.0 * (left_score - right_score)
        estimates[start:stop] = differences
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return ConfidenceInterval(
        estimate,
        float(low),
        float(high),
        confidence,
        resamples,
        seed,
        "mme_image_group",
    )
