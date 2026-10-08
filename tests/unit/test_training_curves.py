import json
from pathlib import Path

import pytest

from invllava.analysis.training_curves import (
    _points,
    _scale_label,
    audit_metric_accounting,
    save_training_curves,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _accounting_fixture(root, label, version, *, complete=True):
    run = root / label
    checkpoint = run / "checkpoints/step-0000002"
    checkpoint.mkdir(parents=True)
    if complete:
        (checkpoint / "COMPLETE").write_text("complete\n")
    payload = {"experiment_id": label, "total_steps": 4}
    if version is not None:
        payload["metric_token_accounting"] = version
    (checkpoint / "metadata.json").write_text(json.dumps(payload))
    return run / "metrics.jsonl"


def test_accounting_preserves_source_metadata_hashes(tmp_path):
    path = _accounting_fixture(tmp_path, "current", "expanded-causal-v2")
    audit = audit_metric_accounting({"current": path}, required=True)
    assert audit["current"]["metric_token_accounting"] == "expanded-causal-v2"
    assert len(audit["current"]["checkpoint_metadata"][0]["sha256"]) == 64


def test_accounting_identifies_retained_earlier_definition(tmp_path):
    path = _accounting_fixture(tmp_path, "earlier", None)
    audit = audit_metric_accounting({"earlier": path}, required=True)
    assert audit["earlier"]["metric_token_accounting"] == "pre-expansion-v1"


def test_accounting_rejects_mixed_log_definitions(tmp_path):
    old = _accounting_fixture(tmp_path, "old", None)
    new = _accounting_fixture(tmp_path, "new", "expanded-causal-v2")
    with pytest.raises(ValueError, match="different logged loss accounting"):
        audit_metric_accounting({"old": old, "new": new}, required=True)


def test_accounting_rejects_incomplete_proof_for_matched_overlay(tmp_path):
    path = _accounting_fixture(tmp_path, "incomplete", "expanded-causal-v2", complete=False)
    with pytest.raises(ValueError, match="complete-checkpoint metadata"):
        audit_metric_accounting({"incomplete": path}, required=True)
    assert (
        audit_metric_accounting({"incomplete": path})["incomplete"]["metric_token_accounting"]
        is None
    )


def test_accounting_rejects_unrecognized_version(tmp_path):
    path = _accounting_fixture(tmp_path, "unknown", "undeclared")
    with pytest.raises(ValueError, match="unrecognized"):
        audit_metric_accounting({"unknown": path})


def test_accounting_rejects_changed_definition_within_run(tmp_path):
    path = _accounting_fixture(tmp_path, "mixed", None)
    second = path.parent / "checkpoints/step-0000004"
    second.mkdir()
    (second / "COMPLETE").write_text("complete\n")
    (second / "metadata.json").write_text(
        json.dumps(
            {
                "experiment_id": "mixed",
                "total_steps": 4,
                "metric_token_accounting": "expanded-causal-v2",
            }
        )
    )
    with pytest.raises(ValueError, match="mixed loss accounting"):
        audit_metric_accounting({"mixed": path})


@pytest.mark.parametrize("payload", [{}, [], {"experiment_id": "x", "total_steps": True}])
def test_accounting_rejects_unrecognized_metadata(tmp_path, payload):
    path = _accounting_fixture(tmp_path, "invalid", None)
    (path.parent / "checkpoints/step-0000002/metadata.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unrecognized native checkpoint metadata"):
        audit_metric_accounting({"invalid": path})


@pytest.mark.parametrize("layer,component", [(0, "q"), (16, "k"), (31, "v")])
def test_native_scale_label(layer: int, component: str) -> None:
    key = f"fusion_scale/language_model.model.layers.{layer}.self_attn.fusion.scales.{component}"
    assert _scale_label(key) == f"L{layer}/{component.upper()}"


def test_unknown_scale_label_is_preserved() -> None:
    assert _scale_label("fusion_scale/custom.module.scale") == "custom.module.scale"


def test_training_plot_retains_extremes_and_full_keys(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    scale_key = "fusion_scale/language_model.model.layers.16.self_attn.fusion.scales.q"
    records = [
        {"step": 1, "train/loss": 2.0, "train/gradient_norm": 1200.0, scale_key: 1.0},
        {"step": 2, "train/loss": 0.4, "train/gradient_norm": 0.0, scale_key: 0.99},
    ]
    closed = []
    original_close = plt.close

    def record_close(figure=None):
        if hasattr(figure, "axes"):
            closed.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", record_close)
    destination = tmp_path / "curves.pdf"
    result = save_training_curves({"layer16": records}, destination)
    assert destination.read_bytes().startswith(b"%PDF")
    assert b"/CreationDate" not in destination.read_bytes()
    assert result["smoothing"] == "none"
    assert result["gradient_norm_axis"] == {"scale": "symlog", "linthresh": 0.1}
    assert scale_key in result["series"]["layer16"]["metric_keys"]
    assert len(closed) == 1
    gradient_axis = closed[0].axes[2]
    assert gradient_axis.get_yscale() == "symlog"
    assert list(gradient_axis.lines[0].get_ydata()) == [1200.0, 0.0]
    assert closed[0].axes[5].lines[0].get_label() == "layer16:L16/Q"


@pytest.mark.parametrize("histories", [{}, {"empty": []}])
def test_training_plot_requires_data(histories, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        save_training_curves(histories, tmp_path / "empty.pdf")


def test_training_points_reject_nonfinite_and_skip_missing() -> None:
    assert _points([{"step": 1}, {"step": 2, "x": 3.0}], "x") == ([2], [3.0])
    with pytest.raises(ValueError, match="non-finite at step 2"):
        _points([{"step": 2, "x": float("nan")}], "x")


def test_model_colors_are_consistent_across_panels(tmp_path: Path, monkeypatch) -> None:
    import matplotlib.pyplot as plt

    figures = []
    original_close = plt.close

    def capture_close(figure=None):
        if hasattr(figure, "axes"):
            figures.append(figure)
        original_close(figure)

    monkeypatch.setattr(plt, "close", capture_close)
    history = [
        {
            "step": 1,
            "train/loss": 0.5,
            "train/peak_allocated_bytes": 1024,
            "train/peak_reserved_bytes": 2048,
            "fusion_scale/language_model.model.layers.0.self_attn.fusion.scales.q": 1.0,
        }
    ]
    metadata = save_training_curves({"parent": history, "replay": history}, tmp_path / "colors.pdf")
    axes = figures[0].axes
    for index, label in enumerate(("parent", "replay")):
        color = metadata["series_colors"][label]
        assert axes[0].lines[index].get_color() == color
        assert axes[4].lines[2 * index].get_color() == color
        assert axes[4].lines[2 * index + 1].get_color() == color
        assert axes[5].lines[index].get_color() == color
        assert axes[4].lines[2 * index].get_linestyle() == "-"
        assert axes[4].lines[2 * index + 1].get_linestyle() == "--"


def _matched_history():
    return [
        {
            "step": step,
            "train/loss": 1.0 / step,
            "train/learning_rate": 0.0002,
            "train/samples_seen": 128 * step,
            "train/tokens_seen": 15000 * step,
            "train/supervised_tokens_in_update": 1200,
        }
        for step in (1, 2)
    ]


def test_compact_loss_view_preserves_raw_points(tmp_path: Path) -> None:
    metadata = save_training_curves(
        {"Layer 0": _matched_history(), "Layer 16": _matched_history()},
        tmp_path / "loss.pdf",
        view="loss",
        require_matched_exposure=True,
    )
    assert metadata["view"] == "loss"
    assert metadata["smoothing"] == "none"
    assert len(metadata["matched_exposure_fields"]) == 4
    assert metadata["series"]["Layer 0"]["record_count"] == 2


@pytest.mark.parametrize(
    "key",
    [
        "train/samples_seen",
        "train/tokens_seen",
        "train/supervised_tokens_in_update",
        "train/learning_rate",
    ],
)
def test_loss_overlay_rejects_different_exposure(tmp_path: Path, key: str) -> None:
    altered = _matched_history()
    altered[1][key] += 1
    with pytest.raises(ValueError, match="unmatched"):
        save_training_curves(
            {"reference": _matched_history(), "altered": altered},
            tmp_path / "loss.pdf",
            view="loss",
            require_matched_exposure=True,
        )
    assert not (tmp_path / "loss.pdf").exists()


def test_loss_overlay_rejects_missing_steps_and_metrics(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identical recorded steps"):
        save_training_curves(
            {"full": _matched_history(), "partial": _matched_history()[:1]},
            tmp_path / "loss.pdf",
            require_matched_exposure=True,
        )
    missing = _matched_history()
    del missing[0]["train/loss"]
    with pytest.raises(ValueError, match="requires train/loss"):
        save_training_curves({"missing": missing}, tmp_path / "loss.pdf", view="loss")


@pytest.mark.parametrize("steps", [[1, 1], [2, 1], [0, 1], [True, 2], [1.5, 2]])
def test_training_view_rejects_invalid_steps(tmp_path: Path, steps) -> None:
    history = _matched_history()
    for row, step in zip(history, steps, strict=True):
        row["step"] = step
    with pytest.raises(ValueError, match="strictly increasing"):
        save_training_curves({"invalid": history}, tmp_path / "loss.pdf")
