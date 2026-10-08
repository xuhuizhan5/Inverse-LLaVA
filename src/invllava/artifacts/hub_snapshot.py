"""Pinned, resumable Hugging Face snapshots with byte-level local evidence."""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file

_CONTROL_FILES = {"COMPLETE", "snapshot-manifest.json"}


def _included(filename: str, allow_patterns: tuple[str, ...]) -> bool:
    return not allow_patterns or any(
        fnmatch.fnmatch(filename, pattern) for pattern in allow_patterns
    )


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"snapshot contains a symlink: {relative}")
        if (
            not path.is_file()
            or relative.parts[0] == ".cache"
            or relative.as_posix() in _CONTROL_FILES
        ):
            continue
        records.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ValueError(f"snapshot contains no files: {root}")
    return records


def verify_hub_snapshot(path: str | Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir() or not (root / "COMPLETE").is_file():
        raise ValueError(f"incomplete Hugging Face snapshot: {root}")
    manifest = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("files") != _inventory(root):
        raise ValueError("Hugging Face snapshot inventory does not match local bytes")
    return manifest


def _require_matching_request(
    record: dict[str, Any],
    *,
    repo_id: str,
    revision: str,
    repo_type: str,
    allow_patterns: tuple[str, ...],
) -> None:
    resolved = record.get("resolved_revision", "")
    if not isinstance(resolved, str) or not re.fullmatch(r"[a-fA-F0-9]{40}", resolved):
        raise ValueError("snapshot has no immutable resolved Hub revision")
    if (
        record.get("repo_id") != repo_id
        or record.get("repo_type") != repo_type
        or revision not in {record.get("requested_revision"), resolved}
        or tuple(record.get("allow_patterns", ())) != allow_patterns
    ):
        raise ValueError("snapshot belongs to a different Hub source or file selection")


def acquire_hub_snapshot(
    *,
    repo_id: str,
    revision: str,
    destination: str | Path,
    repo_type: str = "model",
    allow_patterns: tuple[str, ...] = (),
    minimum_free_bytes_after: int = 50 * 1024**3,
) -> Path:
    """Resolve once, resume, hash, and seal a snapshot.

    An existing destination retains its first resolved commit, including when
    the request originally named a branch. Use a new destination for an update.
    """

    if repo_type not in {"model", "dataset"}:
        raise ValueError("repo_type must be model or dataset")
    if not repo_id or not revision:
        raise ValueError("repo_id and revision are required")
    if minimum_free_bytes_after < 0:
        raise ValueError("minimum free bytes must be non-negative")
    target = Path(destination).resolve()
    if target.exists():
        record = verify_hub_snapshot(target)
        _require_matching_request(
            record,
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
        )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    acquisition = target.with_name(target.name + ".acquisition.json")
    if partial.exists() and not acquisition.is_file():
        raise ValueError("partial snapshot lacks its acquisition record; inspect before resuming")

    from huggingface_hub import HfApi, snapshot_download

    if acquisition.is_file():
        record = json.loads(acquisition.read_text(encoding="utf-8"))
        _require_matching_request(
            record,
            repo_id=repo_id,
            revision=revision,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
        )
        resolved_revision = str(record["resolved_revision"])
        expected_bytes = record.get("selected_metadata_bytes")
    else:
        api = HfApi()
        info = (
            api.model_info(repo_id, revision=revision, files_metadata=True)
            if repo_type == "model"
            else api.dataset_info(repo_id, revision=revision, files_metadata=True)
        )
        resolved_revision = str(info.sha)
        if not re.fullmatch(r"[a-fA-F0-9]{40}", resolved_revision):
            raise ValueError("Hub did not return an immutable commit SHA")
        selected_sizes = [
            sibling.size
            for sibling in info.siblings
            if _included(sibling.rfilename, allow_patterns) and sibling.size is not None
        ]
        expected_bytes = sum(selected_sizes) if selected_sizes else None
        atomic_write_json(
            acquisition,
            {
                "schema_version": 1,
                "repo_id": repo_id,
                "repo_type": repo_type,
                "requested_revision": revision,
                "resolved_revision": resolved_revision,
                "allow_patterns": list(allow_patterns),
                "selected_metadata_bytes": expected_bytes,
            },
        )
    free = shutil.disk_usage(target.parent).free
    remaining_estimate = max(0, int(expected_bytes or 0) - _present_bytes(partial))
    if remaining_estimate + minimum_free_bytes_after > free:
        raise OSError(
            f"snapshot needs an estimated {remaining_estimate} bytes plus "
            f"{minimum_free_bytes_after} bytes reserved headroom, but only {free} bytes are free"
        )
    partial.mkdir(exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved_revision,
        local_dir=partial,
        allow_patterns=list(allow_patterns) or None,
    )
    files = _inventory(partial)
    atomic_write_json(
        partial / "snapshot-manifest.json",
        {
            "format": "invllava-huggingface-snapshot-v1",
            "repo_id": repo_id,
            "repo_type": repo_type,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "allow_patterns": list(allow_patterns),
            "selected_metadata_bytes": expected_bytes,
            "total_bytes": sum(int(item["size_bytes"]) for item in files),
            "files": files,
        },
    )
    atomic_write_text(partial / "COMPLETE", "complete\n")
    os.replace(partial, target)
    acquisition.unlink()
    return target


def _present_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(
        item.stat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink()
    )
