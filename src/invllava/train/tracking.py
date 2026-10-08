from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from invllava.artifacts.atomic import atomic_write_json
from invllava.train.metrics import JsonlMetricWriter


class Tracker(Protocol):
    def log(self, values: dict[str, Any], step: int) -> None: ...

    def close(self) -> None: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalTracker:
    """Canonical append-only metrics with optional, non-authoritative mirrors."""

    def __init__(self, run_dir: str | Path, *, backend: str = "local") -> None:
        self.run_dir = Path(run_dir)
        self.writer = JsonlMetricWriter(self.run_dir / "metrics.jsonl")
        self.metadata_path = self.run_dir / "tracking.json"
        self.metadata: dict[str, Any] = {
            "schema_version": 1,
            "canonical_metrics": "metrics.jsonl",
            "mirror": backend,
            "status": "active",
            "started_at": _utc_now(),
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        atomic_write_json(self.metadata_path, self.metadata)

    def log(self, values: dict[str, Any], step: int) -> None:
        self.writer.write({"step": step, **values})

    def close(self) -> None:
        self.metadata["status"] = "closed"
        self.metadata["closed_at"] = _utc_now()
        self._write_metadata()


class NullTracker:
    def log(self, values: dict[str, Any], step: int) -> None:
        return None

    def close(self) -> None:
        return None


class TensorBoardTracker(LocalTracker):
    def __init__(self, run_dir: str | Path) -> None:
        super().__init__(run_dir, backend="tensorboard")
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "TensorBoard tracking requires the 'tracking' optional dependency"
            ) from error

        self.tensorboard = SummaryWriter(Path(run_dir) / "tensorboard")
        self.metadata["mirror_path"] = "tensorboard"
        self._write_metadata()

    def log(self, values: dict[str, Any], step: int) -> None:
        super().log(values, step)
        for key, value in values.items():
            if isinstance(value, (int, float)):
                self.tensorboard.add_scalar(key, value, step)

    def close(self) -> None:
        self.tensorboard.close()
        super().close()


class WandbTracker(LocalTracker):
    """W&B mirror that is offline unless the operator explicitly opts online.

    W&B never owns checkpoints or canonical metrics. This prevents a login or
    machine-level W&B setting from silently changing the persistence contract.
    """

    def __init__(self, run_dir: str | Path, *, run_id: str, config: dict[str, Any]) -> None:
        super().__init__(run_dir, backend="wandb")
        try:
            import wandb
        except ModuleNotFoundError as error:
            raise RuntimeError("W&B tracking requires the 'wandb' optional dependency") from error

        mode = os.environ.get("WANDB_MODE", "offline").strip().lower()
        if mode not in {"offline", "online"}:
            raise ValueError("WANDB_MODE must be explicitly 'offline' or 'online'")
        project = os.environ.get("WANDB_PROJECT", "").strip()
        if mode == "online" and not project:
            raise ValueError("online W&B tracking requires an explicit WANDB_PROJECT")
        project = project or "inverse-llava"
        entity = os.environ.get("WANDB_ENTITY", "").strip() or None
        group = os.environ.get("WANDB_RUN_GROUP", "").strip() or None
        tags_value = config.get("tags", ())
        tags = [str(value) for value in tags_value] if isinstance(tags_value, (list, tuple)) else []

        self.run = wandb.init(
            id=run_id,
            name=run_id,
            project=project,
            entity=entity,
            group=group,
            tags=tags,
            resume="allow",
            mode=mode,
            dir=str(run_dir),
            config=config,
            save_code=False,
        )
        self.metadata.update(
            {
                "mode": mode,
                "project": project,
                "entity": entity,
                "group": group,
                "run_id": run_id,
                "run_url": getattr(self.run, "url", None) if mode == "online" else None,
                "checkpoint_upload": False,
            }
        )
        self._write_metadata()

    def log(self, values: dict[str, Any], step: int) -> None:
        super().log(values, step)
        self.run.log(values, step=step)

    def close(self) -> None:
        self.run.finish()
        super().close()


def make_tracker(kind: str, run_dir: str | Path, *, run_id: str, config: dict[str, Any]) -> Tracker:
    if kind == "local":
        return LocalTracker(run_dir)
    if kind == "tensorboard":
        return TensorBoardTracker(run_dir)
    if kind == "wandb":
        return WandbTracker(run_dir, run_id=run_id, config=config)
    raise ValueError(f"unsupported tracker: {kind}")
