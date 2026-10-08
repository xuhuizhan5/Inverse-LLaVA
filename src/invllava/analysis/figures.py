from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def joint_pca(series: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit one PCA basis to aligned, equal-width representation series."""

    if len(series) < 2:
        raise ValueError("joint PCA requires at least two labeled series")
    arrays = {label: np.asarray(value, dtype=np.float64) for label, value in series.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes))[0] < 2:
        raise ValueError("joint PCA series must share [samples,features] shape")
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("joint PCA inputs must be finite matrices")
    from sklearn.decomposition import PCA

    labels = list(arrays)
    rows = next(iter(arrays.values())).shape[0]
    pca = PCA(n_components=2, svd_solver="full")
    transformed = pca.fit_transform(np.concatenate([arrays[label] for label in labels], axis=0))
    coordinates = {
        label: transformed[index * rows : (index + 1) * rows] for index, label in enumerate(labels)
    }
    metadata = {
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "sample_count": rows,
        "feature_dim": next(iter(arrays.values())).shape[1],
        "fit_policy": "one full-SVD PCA basis over concatenated labeled series",
    }
    return coordinates, metadata


def save_joint_pca(series: dict[str, np.ndarray], destination: str | Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    coordinates, metadata = joint_pca(series)
    figure, axis = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    for label, points in coordinates.items():
        axis.scatter(points[:, 0], points[:, 1], s=10, alpha=0.55, label=label)
    axis.set(title="Joint PCA (descriptive)", xlabel="PC1", ylabel="PC2")
    axis.legend(frameon=False)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return metadata


def save_cka_rank_panel(
    cka: np.ndarray,
    left_keys: list[str],
    right_keys: list[str],
    left_ranks: list[float],
    right_ranks: list[float],
    destination: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    image = axes[0].imshow(cka, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[0].set(
        title="Linear CKA",
        xlabel="Right representation",
        ylabel="Left representation",
        xticks=np.arange(len(right_keys)),
        yticks=np.arange(len(left_keys)),
        xticklabels=right_keys,
        yticklabels=left_keys,
    )
    axes[0].tick_params(axis="x", labelrotation=70, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=7)
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].plot(left_ranks, marker="o", label="left")
    axes[1].plot(right_ranks, marker="s", label="right")
    axes[1].set(title="Participation ratio", xlabel="Ordered representation", ylabel="Rank")
    axes[1].legend(frameon=False)
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def save_grouped_score_breakdown(
    series: dict[str, dict[str, float]],
    destination: str | Path,
    *,
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    """Plot aligned category scores from two or more sealed score artifacts."""

    if len(series) < 2:
        raise ValueError("score breakdown requires at least two model series")
    categories: list[str] | None = None
    normalized: dict[str, dict[str, float]] = {}
    for label, values in series.items():
        if not values:
            raise ValueError(f"score breakdown series is empty: {label}")
        current = list(values)
        if categories is None:
            categories = current
        elif current != categories:
            raise ValueError("score breakdown series must have identical ordered categories")
        numeric = {key: float(value) for key, value in values.items()}
        if not np.isfinite(list(numeric.values())).all():
            raise ValueError(f"score breakdown contains a non-finite value: {label}")
        normalized[label] = numeric
    assert categories is not None

    import matplotlib.pyplot as plt

    x = np.arange(len(categories), dtype=float)
    width = 0.82 / len(normalized)
    # Keep labels readable when a panel is placed at manuscript text width.
    figure_width = max(6.0, 0.55 * len(categories) + 1.0)
    figure_size = (figure_width, 3.8)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    offset_origin = -0.5 * width * (len(normalized) - 1)
    for index, (label, values) in enumerate(normalized.items()):
        heights = [values[category] for category in categories]
        axis.bar(x + offset_origin + index * width, heights, width=width, label=label)
    axis.set(title=title, ylabel=ylabel, xticks=x, xticklabels=categories)
    axis.tick_params(axis="x", labelrotation=38, labelsize=10)
    axis.legend(frameon=False)
    axis.grid(axis="y", linewidth=0.6, alpha=0.25)
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return {
        "categories": categories,
        "series": normalized,
        "title": title,
        "ylabel": ylabel,
        "figure_size_inches": list(figure_size),
    }


def save_profile_comparison(
    profiles: dict[str, dict[str, Any]], destination: str | Path
) -> dict[str, Any]:
    """Plot comparable portable inference profiles from sealed JSON artifacts."""

    if len(profiles) < 2:
        raise ValueError("profile comparison requires at least two model profiles")
    contracts = {
        (
            payload.get("hardware"),
            payload.get("dtype"),
            payload.get("attention_backend"),
            payload.get("fixed_decode_policy"),
        )
        for payload in profiles.values()
    }
    if len(contracts) != 1:
        raise ValueError("profile comparison requires one hardware and runtime contract")
    batch_sets = [
        tuple(sorted(payload.get("profiles", {}), key=int)) for payload in profiles.values()
    ]
    if not batch_sets[0] or any(batches != batch_sets[0] for batches in batch_sets[1:]):
        raise ValueError("profile comparison requires identical batch sizes")

    batches = batch_sets[0]
    metrics = (
        ("median_decode_tokens_per_second", "Decode tokens/s", 1.0),
        ("median_time_to_first_token_seconds", "Time to first token (ms)", 1000.0),
        ("peak_allocated_bytes", "Peak allocated memory (GiB)", 1.0 / (1024**3)),
    )
    normalized: dict[str, dict[str, dict[str, float]]] = {}
    for label, payload in profiles.items():
        values: dict[str, dict[str, float]] = {}
        for batch in batches:
            autoregressive = payload["profiles"][batch].get("autoregressive")
            if not isinstance(autoregressive, dict):
                raise ValueError(f"profile lacks autoregressive batch {batch}: {label}")
            values[batch] = {}
            for key, _, _scale in metrics:
                value = float(autoregressive[key])
                if not np.isfinite(value):
                    raise ValueError(f"profile contains non-finite {key}: {label}")
                values[batch][key] = value
        normalized[label] = values

    import matplotlib.pyplot as plt

    x = np.arange(len(batches), dtype=float)
    width = 0.8 / len(profiles)
    # Keep text legible at journal width and embed searchable TrueType fonts.
    # The local context does not change the caller's other figures.
    with plt.rc_context({"font.size": 14, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        offset_origin = -0.5 * width * (len(profiles) - 1)
        for axis, (metric, title, scale) in zip(axes, metrics, strict=True):
            for index, (label, values) in enumerate(normalized.items()):
                axis.bar(
                    x + offset_origin + index * width,
                    [values[batch][metric] * scale for batch in batches],
                    width=width,
                    label=label,
                )
            axis.set(title=title, xlabel="Batch size", xticks=x, xticklabels=batches)
            axis.grid(axis="y", linewidth=0.6, alpha=0.25)
        axes[0].legend(frameon=False)
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if Path(destination).suffix.lower() == ".pdf"
            else None
        )
        figure.savefig(destination, dpi=300, metadata=metadata)
        plt.close(figure)
    hardware, dtype, attention_backend, decode_policy = next(iter(contracts))
    return {
        "hardware": hardware,
        "dtype": dtype,
        "attention_backend": attention_backend,
        "fixed_decode_policy": decode_policy,
        "batch_sizes": [int(batch) for batch in batches],
        "series": normalized,
        "series_units": {
            "median_decode_tokens_per_second": "tokens/second",
            "median_time_to_first_token_seconds": "seconds",
            "peak_allocated_bytes": "bytes",
        },
        "parameter_counts": {
            label: {
                "total": int(payload["total_parameters"]),
                "training_trainable": (
                    int(payload["trainable_parameters"])
                    if payload.get("trainable_parameters") is not None
                    else None
                ),
                "runtime_requires_grad": (
                    int(payload["runtime_requires_grad_parameters"])
                    if payload.get("runtime_requires_grad_parameters") is not None
                    else None
                ),
            }
            for label, payload in profiles.items()
        },
    }
