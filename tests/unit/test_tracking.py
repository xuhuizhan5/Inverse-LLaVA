from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from invllava.train.tracking import LocalTracker, WandbTracker


def test_local_tracker_records_canonical_metrics_and_metadata(tmp_path: Path) -> None:
    tracker = LocalTracker(tmp_path)
    tracker.log({"train/loss": 1.25}, step=1)
    tracker.close()

    metric = json.loads((tmp_path / "metrics.jsonl").read_text(encoding="utf-8"))
    metadata = json.loads((tmp_path / "tracking.json").read_text(encoding="utf-8"))
    assert metric == {"step": 1, "train/loss": 1.25}
    assert metadata["mirror"] == "local"
    assert metadata["status"] == "closed"


def test_wandb_defaults_offline_and_never_uploads_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class Run:
        url = None

        def log(self, values: dict[str, object], *, step: int) -> None:
            calls.append({"values": values, "step": step})

        def finish(self) -> None:
            calls.append({"finished": True})

    def init(**kwargs: object) -> Run:
        calls.append(kwargs)
        return Run()

    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    tracker = WandbTracker(tmp_path, run_id="fixture", config={"tags": ["canary"]})
    tracker.log({"train/loss": 0.5}, step=1)
    tracker.close()

    assert calls[0]["mode"] == "offline"
    assert calls[0]["project"] == "inverse-llava"
    assert calls[0]["save_code"] is False
    metadata = json.loads((tmp_path / "tracking.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_upload"] is False
    assert metadata["mode"] == "offline"


def test_wandb_online_requires_explicit_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **_: None))
    with pytest.raises(ValueError, match="WANDB_PROJECT"):
        WandbTracker(tmp_path, run_id="fixture", config={})
