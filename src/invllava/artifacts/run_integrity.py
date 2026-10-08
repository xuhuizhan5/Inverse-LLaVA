"""Integrity checks for completed training runs and restored backups."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.manifest import RunManifest
from invllava.train.checkpoint import checkpoint_inventory
from invllava.train.metrics import read_metric_history


@dataclass(frozen=True)
class RunIntegrityReport:
    schema_version: int
    run_dir: str
    run_id: str
    complete: bool
    artifact_count: int
    checkpoint_count: int
    metric_records: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest artifact is outside the run: {relative_path}")
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"manifest artifact must not be a symlink: {relative_path}")
    path = candidate.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"manifest artifact escapes the run: {relative_path}")
    return path


def verify_training_run(
    run_dir: str | Path,
    *,
    require_complete: bool = True,
) -> RunIntegrityReport:
    root = Path(run_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(root)
    manifest = RunManifest.read(root / "manifest.json")
    if manifest.run_id != root.name:
        raise ValueError(f"manifest run_id {manifest.run_id!r} does not match {root.name!r}")

    complete = (root / "RUN_COMPLETE").is_file()
    if require_complete and not complete:
        raise ValueError("run has no RUN_COMPLETE marker")
    relative_paths = [record.relative_path for record in manifest.artifacts]
    if len(relative_paths) != len(set(relative_paths)):
        raise ValueError("manifest contains duplicate artifact paths")
    for record in manifest.artifacts:
        path = _resolve_artifact(root, record.relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if record.size_bytes is not None and path.stat().st_size != record.size_bytes:
            raise ValueError(f"artifact size mismatch: {record.relative_path}")
        if record.sha256 is not None and sha256_file(path) != record.sha256:
            raise ValueError(f"artifact digest mismatch: {record.relative_path}")

    inventory_path = root / "checkpoint_inventory.json"
    checkpoint_count = 0
    if inventory_path.is_file():
        recorded = json.loads(inventory_path.read_text(encoding="utf-8"))
        observed = checkpoint_inventory(root / "checkpoints")
        if recorded != observed:
            raise ValueError("checkpoint inventory does not match checkpoint bytes")
        checkpoint_count = len(observed["checkpoints"])
    elif require_complete:
        raise FileNotFoundError(inventory_path)

    metrics_path = root / "metrics.jsonl"
    metric_records = len(read_metric_history(metrics_path)) if metrics_path.is_file() else 0
    return RunIntegrityReport(
        schema_version=1,
        run_dir=str(root),
        run_id=manifest.run_id,
        complete=complete,
        artifact_count=len(manifest.artifacts),
        checkpoint_count=checkpoint_count,
        metric_records=metric_records,
    )
