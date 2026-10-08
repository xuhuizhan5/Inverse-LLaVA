"""Training diagnostics and loss overlays from recorded optimizer updates."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from invllava.analysis.plot_style import CATEGORICAL


def audit_metric_accounting(
    sources: dict[str, str | Path], *, required: bool = False
) -> dict[str, Any]:
    """Bind logged loss definitions to retained complete-checkpoint metadata."""
    from invllava.artifacts.hashing import sha256_file

    records = {}
    for label, metrics in sources.items():
        evidence = []
        versions = set()
        for path in sorted((Path(metrics).parent / "checkpoints").glob("step-*/metadata.json")):
            if not (path.parent / "COMPLETE").is_file():
                continue
            payload = json.loads(path.read_text())
            if (
                not isinstance(payload, dict)
                or not isinstance(payload.get("experiment_id"), str)
                or not payload["experiment_id"]
                or type(payload.get("total_steps")) is not int
                or payload["total_steps"] < 1
            ):
                raise ValueError(f"{label}: unrecognized native checkpoint metadata: {path}")
            version = payload.get("metric_token_accounting", "pre-expansion-v1")
            if version not in {"pre-expansion-v1", "expanded-causal-v2"}:
                raise ValueError(f"{label}: unrecognized metric-token accounting: {version}")
            versions.add(version)
            evidence.append({"path": str(path), "sha256": sha256_file(path)})
        if len(versions) > 1:
            raise ValueError(f"{label}: checkpoint metadata contains mixed loss accounting")
        version = next(iter(versions), None)
        if required and version is None:
            raise ValueError(
                f"{label}: matched curves require complete-checkpoint metadata alongside metrics"
            )
        records[label] = {"metric_token_accounting": version, "checkpoint_metadata": evidence}
    known = {item["metric_token_accounting"] for item in records.values()} - {None}
    if len(known) > 1:
        raise ValueError("training curves cannot combine different logged loss accounting")
    return records


def _scale_label(key: str) -> str:
    """Shorten native scale names for display; sidecars retain complete keys."""

    match = re.fullmatch(
        r"fusion_scale/language_model\.model\.layers\.(\d+)\.self_attn\.fusion\.scales\.([qkv])",
        key,
    )
    if match:
        return f"L{match[1]}/{match[2].upper()}"
    return key.removeprefix("fusion_scale/")


def _points(
    records: list[dict[str, Any]],
    key: str,
    *,
    scale: float = 1.0,
) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for record in records:
        value = record.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        numeric = float(value) / scale
        if not math.isfinite(numeric):
            raise ValueError(f"training metric {key} is non-finite at step {record.get('step')}")
        steps.append(int(record["step"]))
        values.append(numeric)
    return steps, values


def save_training_curves(
    histories: dict[str, list[dict[str, Any]]],
    destination: str | Path,
    *,
    view: str = "diagnostics",
    require_matched_exposure: bool = False,
) -> dict[str, Any]:
    """Plot raw metrics; optionally require identical recorded update exposures.

    Matching counters/schedules is a necessary comparability check. The caller
    must also verify selected row IDs, tokenizer, masking, and training recipe.
    Loss describes the training objective and is not an evaluation score.
    """

    if not histories or any(not records for records in histories.values()):
        raise ValueError("training curves require at least one non-empty labeled history")
    if view not in {"diagnostics", "loss"}:
        raise ValueError(f"unknown training-curve view: {view}")
    for label, records in histories.items():
        previous = 0
        for record in records:
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool) or step <= previous:
                raise ValueError(f"{label}: steps must be strictly increasing positive integers")
            previous = step
    exposure_fields = (
        "train/samples_seen",
        "train/tokens_seen",
        "train/supervised_tokens_in_update",
        "train/learning_rate",
    )
    if require_matched_exposure:
        reference = next(iter(histories.values()))
        for label, records in histories.items():
            if [row["step"] for row in records] != [row["step"] for row in reference]:
                raise ValueError(f"{label}: matched overlays require identical recorded steps")
            for expected, actual in zip(reference, records, strict=True):
                for key in exposure_fields:
                    value = actual.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                        or value != expected.get(key)
                    ):
                        raise ValueError(f"{label}: unmatched {key} at step {actual['step']}")
    if view == "loss":
        for label, records in histories.items():
            for key in ("train/loss", "train/learning_rate"):
                if len(_points(records, key)[0]) != len(records):
                    raise ValueError(f"{label}: loss view requires {key} at every recorded step")
    import matplotlib.pyplot as plt

    shape, size = ((1, 2), (8.0, 3.0)) if view == "loss" else ((2, 3), (12.5, 7.2))
    figure, axes = plt.subplots(*shape, figsize=size, constrained_layout=True)
    palette = (*CATEGORICAL, "#CC79A7", "#56B4E9", "#E69F00")
    colors = {label: palette[index % len(palette)] for index, label in enumerate(histories)}
    line_styles = {
        label: ("-", "--", "-.", ":")[index % 4] for index, label in enumerate(histories)
    }
    scale_styles = {"q": "-", "k": "--", "v": ":"}
    ordinary = (
        ("train/loss", "Supervised loss", 1.0),
        ("train/learning_rate", "Learning rate", 1.0),
        ("train/gradient_norm", "Gradient norm (symlog)", 1.0),
        ("train/examples_per_second", "Examples / second", 1.0),
    )
    panel_count = 2 if view == "loss" else 4
    for axis, (key, title, scale) in zip(
        axes.flat[:panel_count], ordinary[:panel_count], strict=True
    ):
        for label, records in histories.items():
            steps, values = _points(records, key, scale=scale)
            if steps:
                axis.plot(
                    steps,
                    values,
                    color=colors[label],
                    linestyle=line_styles[label],
                    linewidth=1.4,
                    label=label,
                )
        axis.set(title=title, xlabel="Optimizer update")
        if key == "train/gradient_norm":
            # Keep zeros and initial spikes visible without clipping or smoothing.
            axis.set_yscale("symlog", linthresh=0.1)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(frameon=False, fontsize=8)

    if view == "diagnostics":
        _plot_memory_and_scales(axes, histories, colors, scale_styles)

    for axis in axes.flat:
        if not axis.lines:
            axis.text(
                0.5,
                0.5,
                "not recorded",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#64748b",
            )
    metadata = (
        {"CreationDate": None, "ModDate": None}
        if Path(destination).suffix.lower() == ".pdf"
        else None
    )
    try:
        with plt.rc_context({"pdf.fonttype": 42}):
            figure.savefig(destination, dpi=300, metadata=metadata)
    finally:
        plt.close(figure)
    return {
        "view": view,
        "smoothing": "none",
        "series_colors": colors,
        "series_line_styles": line_styles,
        "fusion_scale_line_styles": scale_styles,
        "x_axis": "optimizer_update",
        "matched_exposure_fields": list(exposure_fields) if require_matched_exposure else [],
        "gradient_norm_axis": {"scale": "symlog", "linthresh": 0.1},
        "series": {
            label: {
                "record_count": len(records),
                "first_step": records[0]["step"],
                "last_step": records[-1]["step"],
                "metric_keys": sorted({key for record in records for key in record}),
            }
            for label, records in histories.items()
        },
    }


def _plot_memory_and_scales(axes, histories, colors, scale_styles) -> None:
    """Keep diagnostic-only panels out of the compact loss view."""

    memory_axis = axes.flat[4]
    for label, records in histories.items():
        for key, style, suffix in (
            ("train/peak_allocated_bytes", "-", "allocated"),
            ("train/peak_reserved_bytes", "--", "reserved"),
        ):
            steps, values = _points(records, key, scale=float(1024**3))
            if steps:
                memory_axis.plot(
                    steps,
                    values,
                    linestyle=style,
                    color=colors[label],
                    linewidth=1.3,
                    label=f"{label} {suffix}",
                )
    memory_axis.set(title="Peak GPU memory", xlabel="Optimizer update", ylabel="GiB")
    memory_axis.grid(alpha=0.2)
    if memory_axis.lines:
        memory_axis.legend(frameon=False, fontsize=7)

    scale_axis = axes.flat[5]
    for label, records in histories.items():
        scale_keys = sorted(
            {key for record in records for key in record if key.startswith("fusion_scale/")}
        )
        for key in scale_keys:
            steps, values = _points(records, key)
            if steps:
                scale_axis.plot(
                    steps,
                    values,
                    linewidth=1.3,
                    color=colors[label],
                    linestyle=scale_styles.get(key.rsplit(".", 1)[-1], "-"),
                    label=f"{label}:{_scale_label(key)}",
                )
    scale_axis.set(title="Fusion scales", xlabel="Optimizer update")
    scale_axis.grid(alpha=0.2)
    if scale_axis.lines:
        scale_axis.legend(frameon=False, fontsize=7, ncol=2)
