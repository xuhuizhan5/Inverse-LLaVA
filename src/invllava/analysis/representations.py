from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("representation must be [samples,features] with at least two samples")
    if not np.isfinite(value).all():
        raise ValueError("representation contains NaN or infinity")
    return value


def linear_cka(left: np.ndarray, right: np.ndarray, *, debiased: bool = False) -> float:
    """Linear CKA, optionally normalized from unbiased HSIC estimates.

    The optional U-centered estimator follows Kornblith et al. (2019). Its
    normalized ratio can still be biased and can be negative. Default CKA and
    its existing uncertainty analysis are unchanged.
    """

    left = _matrix(left)
    right = _matrix(right)
    if left.shape[0] != right.shape[0]:
        raise ValueError("CKA representations must be sample-aligned")
    if debiased and left.shape[0] < 4:
        raise ValueError("unbiased HSIC requires at least four aligned samples")
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    # CKA is invariant to a scalar rescaling of either representation. Scaling
    # before the products prevents genuine overflow for large activations. Some
    # NumPy 2.0/Accelerate builds also emit false floating-point flags from BLAS;
    # finite inputs and outputs are checked explicitly around that call.
    left /= max(float(np.abs(left).max()), np.finfo(np.float64).tiny)
    right /= max(float(np.abs(right).max()), np.finfo(np.float64).tiny)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        samples = left.shape[0]
        feature_product_cost = samples * left.shape[1] * right.shape[1]
        gram_product_cost = samples**2 * (left.shape[1] + right.shape[1])
        if debiased or gram_product_cost < feature_product_cost:
            left_gram = left @ left.T
            right_gram = right @ right.T
            if debiased:
                left_gram = _u_center_gram(left_gram)
                right_gram = _u_center_gram(right_gram)
            cross = float(np.sum(left_gram * right_gram))
            denominator = np.linalg.norm(left_gram, ord="fro") * np.linalg.norm(
                right_gram, ord="fro"
            )
        else:
            cross = np.linalg.norm(left.T @ right, ord="fro") ** 2
            denominator = np.linalg.norm(left.T @ left, ord="fro") * np.linalg.norm(
                right.T @ right, ord="fro"
            )
    if not np.isfinite(cross) or not np.isfinite(denominator):
        raise FloatingPointError("linear CKA products are non-finite")
    if not denominator:
        raise ValueError("CKA is undefined for a constant representation")
    return float(cross / denominator)


def _u_center_gram(gram: np.ndarray) -> np.ndarray:
    """U-center a Gram matrix; diagonal self-similarities do not enter HSIC."""
    samples = len(gram)
    original_norm = np.linalg.norm(gram)
    centered = gram.copy()
    np.fill_diagonal(centered, 0)
    row_sums = centered.sum(axis=1)
    centered -= (row_sums[:, None] + row_sums[None, :]) / (samples - 2)
    centered += row_sums.sum() / ((samples - 1) * (samples - 2))
    np.fill_diagonal(centered, 0)
    if np.linalg.norm(centered) <= 64 * np.finfo(np.float64).eps * original_norm:
        raise ValueError("CKA is undefined for a degenerate U-centered representation")
    return centered


def select_layer_views(
    arrays: Mapping[str, np.ndarray], prefix: str
) -> tuple[list[str], list[str]]:
    """Select one pooling view and identify sample-constant layers explicitly.

    ``hidden.`` selects ``hidden.0`` but excludes ``hidden.last.0``. The last
    prompt token's embedding is often identical across examples before the
    first decoder block, so its CKA is undefined rather than zero similarity.
    """
    keys = sorted(
        (key for key in arrays if key.startswith(prefix) and key[len(prefix) :].isdigit()),
        key=lambda key: int(key[len(prefix) :]),
    )
    if not keys:
        raise ValueError("CKA prefix selected no indexed layers")
    constant = []
    for key in keys:
        value = _matrix(arrays[key])
        if np.all(value == value[0]):
            constant.append(key)
    usable = [key for key in keys if key not in constant]
    if not usable:
        raise ValueError("CKA has no nonconstant selected layers")
    return usable, constant


def cka_sample_uncertainty(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    """Paired-row percentile interval and sample-permutation null for linear CKA.

    Rows must be independent sampling units. This conditions on the two fitted
    models; it does not estimate training-seed uncertainty. The permutation null
    removes row correspondence, retaining each model's marginal geometry. It
    cannot distinguish visual grounding from common prompts or pretraining.
    Sample Gram matrices keep repeated calculations independent of feature width.
    """
    left, right = _matrix(left), _matrix(right)
    if left.shape[0] != right.shape[0] or len(left) < 4:
        raise ValueError("CKA uncertainty requires at least four aligned independent rows")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must lie in (0, 1)")

    def gram(value):
        centered = value - value.mean(axis=0, keepdims=True)
        centered /= max(float(np.abs(centered).max()), np.finfo(np.float64).tiny)
        with np.errstate(over="ignore", invalid="ignore"):
            result = centered @ centered.T
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite CKA Gram matrix")
        return result

    def similarity(first, second):
        def center(value):
            means = value.mean(axis=0, keepdims=True)
            return value - means - means.T + value.mean()

        first, second = center(first), center(second)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator <= np.finfo(np.float64).tiny:
            raise ValueError("CKA is undefined for a constant representation or resample")
        return float(np.sum(first * second) / denominator)

    first, second = gram(left), gram(right)
    observed = similarity(first, second)
    rng = np.random.default_rng(seed)
    bootstrap, null = [], []
    undefined = 0
    for _ in range(resamples):
        indices = rng.integers(0, len(left), size=len(left))
        if len(np.unique(indices)) < 2:
            undefined += 1
        else:
            try:
                bootstrap.append(
                    similarity(first[np.ix_(indices, indices)], second[np.ix_(indices, indices)])
                )
            except ValueError:
                undefined += 1
        permutation = rng.permutation(len(left))
        null.append(similarity(first, second[np.ix_(permutation, permutation)]))
    if len(bootstrap) < resamples * 0.9:
        raise ValueError("too many undefined bootstrap CKA samples")
    alpha = (1 - confidence) / 2
    low, high = np.quantile(bootstrap, [alpha, 1 - alpha])
    null_low, null_high = np.quantile(null, [alpha, 1 - alpha])
    return {
        "cka": observed,
        "confidence": confidence,
        "ci_low": float(low),
        "ci_high": float(high),
        "null_mean": float(np.mean(null)),
        "null_interval_low": float(null_low),
        "null_interval_high": float(null_high),
        "one_sided_p_value": float(
            (1 + np.count_nonzero(np.asarray(null) >= observed)) / (resamples + 1)
        ),
        "resamples": resamples,
        "undefined_bootstrap_samples": undefined,
        "sample_count": len(left),
        "seed": seed,
        "method": (
            "paired-row percentile bootstrap; row-permutation null; biased centered linear CKA"
        ),
    }


def effective_rank(value: np.ndarray) -> dict[str, float]:
    centered = _matrix(value) - np.asarray(value).mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular**2
    probabilities = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    nonzero = probabilities[probabilities > 0]
    entropy_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum())) if nonzero.size else 0.0
    participation = (
        float(variance.sum() ** 2 / np.square(variance).sum()) if variance.any() else 0.0
    )
    return {"entropy_effective_rank": entropy_rank, "participation_ratio": participation}


def singular_spectrum(value: np.ndarray) -> tuple[float, ...]:
    """Normalized variance spectrum, ordered from largest to smallest component."""

    centered = _matrix(value) - np.asarray(value).mean(axis=0, keepdims=True)
    variance = np.linalg.svd(centered, compute_uv=False) ** 2
    total = variance.sum()
    normalized = variance / total if total else np.zeros_like(variance)
    return tuple(float(item) for item in normalized)


def rsa_spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    left = _matrix(left)
    right = _matrix(right)
    if left.shape[0] != right.shape[0]:
        raise ValueError("RSA representations must be sample-aligned")
    result = spearmanr(pdist(left, metric="cosine"), pdist(right, metric="cosine"))
    return float(result.statistic)


def knn_overlap(left: np.ndarray, right: np.ndarray, *, k: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors

    left = _matrix(left)
    right = _matrix(right)
    if left.shape[0] != right.shape[0] or not 0 < k < left.shape[0]:
        raise ValueError("k must be smaller than aligned sample count")
    # With X=None sklearn already excludes each query's own training row.
    left_neighbors = (
        NearestNeighbors(n_neighbors=k, metric="cosine").fit(left).kneighbors(return_distance=False)
    )
    right_neighbors = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(right)
        .kneighbors(return_distance=False)
    )
    return float(
        np.mean(
            [
                len(set(left_row).intersection(right_row)) / k
                for left_row, right_row in zip(left_neighbors, right_neighbors, strict=True)
            ]
        )
    )


def matched_shuffled_margin(text: np.ndarray, visual: np.ndarray, *, seed: int = 2026) -> float:
    text = _matrix(text)
    visual = _matrix(visual)
    if text.shape != visual.shape:
        raise ValueError("matched margin requires equal sample and feature dimensions")

    def normalize(value: np.ndarray) -> np.ndarray:
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)

    text = normalize(text)
    visual = normalize(visual)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(visual))
    if np.any(permutation == np.arange(len(visual))):
        permutation = np.roll(np.arange(len(visual)), 1)
    matched = np.sum(text * visual, axis=1)
    shuffled = np.sum(text * visual[permutation], axis=1)
    return float(np.mean(matched - shuffled))


def paired_numerical_similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Summarize numerical drift between shape-aligned representations."""

    left = _matrix(left)
    right = _matrix(right)
    if left.shape != right.shape:
        raise ValueError("paired numerical comparison requires identical shapes")
    difference = left - right
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, np.finfo(np.float64).tiny)
    cosine = np.sum(left * right, axis=1) / denominator
    reference_norm = float(np.linalg.norm(right))
    return {
        "mean_paired_cosine": float(cosine.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "relative_frobenius_error": (
            float(np.linalg.norm(difference) / reference_norm)
            if reference_norm
            else float(np.linalg.norm(difference))
        ),
        "max_absolute_error": float(np.abs(difference).max()),
    }


def matched_permutation_test(
    text: np.ndarray,
    visual: np.ndarray,
    *,
    permutations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Test paired cosine similarity against shuffled sample assignments.

    The complete sample-by-sample cosine matrix is computed once. Random
    permutations then provide a deterministic null distribution without
    repeatedly multiplying the high-dimensional feature matrices.
    """

    text = _matrix(text)
    visual = _matrix(visual)
    if text.shape != visual.shape:
        raise ValueError("matched permutation test requires equal sample and feature dimensions")
    if permutations < 1:
        raise ValueError("permutations must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    def normalize(value: np.ndarray) -> np.ndarray:
        return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        similarities = normalize(text) @ normalize(visual).T
    if not np.isfinite(similarities).all():
        raise FloatingPointError("matched permutation cosine products are non-finite")
    matched = float(np.diag(similarities).mean())
    indices = np.arange(len(text))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        permutation = rng.permutation(len(text))
        null[index] = similarities[indices, permutation].mean()

    alpha = (1.0 - confidence) / 2.0
    null_low, null_high = np.quantile(null, (alpha, 1.0 - alpha))
    return {
        "confidence": confidence,
        "matched_cosine_mean": matched,
        "null_cosine_mean": float(null.mean()),
        "null_ci_low": float(null_low),
        "null_ci_high": float(null_high),
        "matched_minus_null_mean": float(matched - null.mean()),
        "one_sided_p_value": float((1 + np.count_nonzero(null >= matched)) / (permutations + 1)),
        "permutations": permutations,
        "seed": seed,
    }


@dataclass(frozen=True)
class RepresentationSummary:
    cka: float
    rsa: float
    knn_overlap: float
    left_effective_rank: dict[str, float]
    right_effective_rank: dict[str, float]
    left_singular_spectrum: tuple[float, ...]
    right_singular_spectrum: tuple[float, ...]


def compare_representations(
    left: np.ndarray, right: np.ndarray, *, k: int = 10
) -> RepresentationSummary:
    return RepresentationSummary(
        cka=linear_cka(left, right),
        rsa=rsa_spearman(left, right),
        knn_overlap=knn_overlap(left, right, k=k),
        left_effective_rank=effective_rank(left),
        right_effective_rank=effective_rank(right),
        left_singular_spectrum=singular_spectrum(left),
        right_singular_spectrum=singular_spectrum(right),
    )
