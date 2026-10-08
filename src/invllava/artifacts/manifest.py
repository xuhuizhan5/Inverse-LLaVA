from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import optional_sha256_environment


def _git_value(args: list[str], root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    relative_path: str
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass
class RunManifest:
    schema_version: int
    run_id: str
    experiment_id: str
    scientific_id: str
    method_revision: str
    created_at: str
    code_commit: str | None
    code_dirty: bool | None
    command: list[str]
    host: dict[str, str]
    source_configs: list[str]
    dataset_revisions: dict[str, str | None]
    checkpoint_inputs: dict[str, str]
    protocol_ids: list[str]
    artifacts: list[ArtifactRecord] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    execution_source_sha256: str | None = None

    def __post_init__(self) -> None:
        value = self.execution_source_sha256
        if value is not None and (
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("execution_source_sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        experiment_id: str,
        scientific_id: str,
        method_revision: str,
        repository_root: str | Path,
        source_configs: list[str],
        dataset_revisions: dict[str, str | None],
        checkpoint_inputs: dict[str, str] | None = None,
        protocol_ids: list[str] | None = None,
    ) -> RunManifest:
        root = Path(repository_root)
        status = _git_value(["status", "--porcelain"], root)
        return cls(
            schema_version=1,
            run_id=run_id,
            experiment_id=experiment_id,
            scientific_id=scientific_id,
            method_revision=method_revision,
            created_at=datetime.now(timezone.utc).isoformat(),
            code_commit=_git_value(["rev-parse", "HEAD"], root),
            code_dirty=None if status is None else bool(status),
            command=list(sys.argv),
            host={
                "hostname": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            source_configs=source_configs,
            dataset_revisions=dataset_revisions,
            checkpoint_inputs=checkpoint_inputs or {},
            protocol_ids=protocol_ids or [],
            execution_source_sha256=optional_sha256_environment("INVLLAVA_EXECUTION_SOURCE_SHA256"),
        )

    def add_artifact(self, record: ArtifactRecord) -> None:
        if any(item.relative_path == record.relative_path for item in self.artifacts):
            raise ValueError(f"artifact already registered: {record.relative_path}")
        self.artifacts.append(record)

    def write(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        data["artifacts"] = [ArtifactRecord(**item) for item in data.get("artifacts", [])]
        if "execution_source_sha256" not in data:
            # Preserve compatibility with manifests written before the field
            # became explicit; prediction jobs stored the same identity in a
            # structured note.
            prefix = "execution_source_sha256="
            data["execution_source_sha256"] = next(
                (
                    note.removeprefix(prefix)
                    for note in data.get("notes", [])
                    if note.startswith(prefix)
                ),
                None,
            )
        return cls(**data)
