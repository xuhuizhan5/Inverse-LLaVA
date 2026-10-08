"""Build one canonical image root from independently acquired source trees."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.config.schema import DataLayoutEntry, DataLayoutSpec

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _safe_child(root: Path, relative: Path) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"layout source path contains a symbolic link: {relative}")
    target = candidate.resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"layout path escapes its root: {relative}")
    return target


def _materialize_directory(source: Path, target: Path) -> dict[str, int]:
    """Validate, inventory, and hardlink one component in a single traversal."""

    files = 0
    bytes_total = 0
    target.mkdir(parents=True)
    pending = [(source, target)]
    while pending:
        source_directory, target_directory = pending.pop()
        with os.scandir(source_directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            source_path = Path(entry.path)
            target_path = target_directory / entry.name
            if entry.is_symlink():
                raise ValueError(f"layout source contains a link or special file: {source_path}")
            if entry.is_dir(follow_symlinks=False):
                target_path.mkdir()
                pending.append((source_path, target_path))
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f"layout source contains a link or special file: {source_path}")
            file_stat = entry.stat(follow_symlinks=False)
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"layout source contains a link or special file: {source_path}")
            _hardlink(source_path, target_path)
            files += 1
            bytes_total += file_stat.st_size
    return {"files": files, "size_bytes": bytes_total}


def _hardlink(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
    """Create one link and fail immediately with actionable ownership context."""

    try:
        os.link(source, target)
    except OSError as error:
        source_path = Path(source)
        source_uid = source_path.stat().st_uid
        effective_uid = os.geteuid() if hasattr(os, "geteuid") else "unknown"
        raise RuntimeError(
            f"cannot hardlink {source_path}: {error.strerror or error}; "
            f"source UID={source_uid}, effective UID={effective_uid}. Keep the component and "
            "destination on one filesystem and run extraction and assembly under the same UID."
        ) from error


def materialize_hardlink_layout(
    spec: DataLayoutSpec,
    *,
    component_root: str | Path,
    destination: str | Path,
    manifest_path: str | Path,
    layout_config_sha256: str,
) -> Path:
    """Atomically materialize a same-filesystem, zero-payload-copy data tree."""

    if _SHA256.fullmatch(layout_config_sha256) is None:
        raise ValueError("layout config SHA-256 must be 64 lowercase hexadecimal characters")
    components = Path(component_root).resolve(strict=True)
    output = Path(destination).resolve()
    manifest = Path(manifest_path).resolve()
    if output == components or output in components.parents or components in output.parents:
        raise ValueError("component root and destination must be separate sibling trees")
    if manifest == output or output in manifest.parents:
        raise ValueError("layout manifest must be stored outside the assembled data tree")
    if output.exists() or manifest.exists():
        raise FileExistsError(output if output.exists() else manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.stat(components).st_dev != os.stat(output.parent).st_dev:
        raise OSError("component root and destination must be on the same filesystem")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))

    def materialize_entry(entry: DataLayoutEntry) -> dict[str, Any]:
        source = _safe_child(components, entry.source)
        if not source.is_dir() or source.is_symlink():
            raise FileNotFoundError(f"layout source directory is missing: {source}")
        target = temporary.joinpath(*entry.destination.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        inventory = _materialize_directory(source, target)
        return {
            "source": entry.source.as_posix(),
            "destination": entry.destination.as_posix(),
            **inventory,
        }

    try:
        with ThreadPoolExecutor(max_workers=min(8, len(spec.entries))) as executor:
            # map preserves configuration order in the manifest while distinct,
            # schema-validated destination trees are built concurrently.
            records = list(executor.map(materialize_entry, spec.entries))
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    try:
        atomic_write_json(
            manifest,
            {
                "schema_version": 1,
                "format": "invllava-hardlink-data-layout-v1",
                "layout_id": spec.id,
                "component_root": str(components),
                "destination": str(output),
                "entries": records,
                "files": sum(record["files"] for record in records),
                "size_bytes": sum(record["size_bytes"] for record in records),
                "layout_config_sha256": layout_config_sha256,
            },
        )
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return output
