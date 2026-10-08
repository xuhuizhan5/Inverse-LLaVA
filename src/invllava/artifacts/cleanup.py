"""Manifest-scoped cleanup; never recursively targets an inferred broad path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from invllava.artifacts.manifest import RunManifest


@dataclass(frozen=True)
class CleanupItem:
    path: Path
    size_bytes: int


def cleanup_plan(manifest_path: str | Path, root: str | Path) -> tuple[CleanupItem, ...]:
    root_path = Path(root).resolve()
    manifest = RunManifest.read(manifest_path)
    items: list[CleanupItem] = []
    for record in manifest.artifacts:
        target = (root_path / record.relative_path).resolve()
        if target == root_path or root_path not in target.parents:
            raise ValueError(f"refusing cleanup target outside session root: {target}")
        if target.is_file() or target.is_symlink():
            items.append(CleanupItem(target, target.stat().st_size))
    return tuple(items)


def execute_cleanup(plan: tuple[CleanupItem, ...], *, confirmed: bool = False) -> int:
    if not confirmed:
        raise PermissionError(
            "cleanup is dry-run by default; pass confirmed=True after reviewing targets"
        )
    removed = 0
    for item in plan:
        item.path.unlink(missing_ok=True)
        removed += item.size_bytes
    return removed
