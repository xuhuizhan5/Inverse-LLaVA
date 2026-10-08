from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    schema_version: int
    protocol_id: str
    experiment_id: str
    checkpoint_id: str
    sample_id: str
    prompt: str
    prediction: str
    references: tuple[str, ...] = ()
    image_ids: tuple[str, ...] = ()
    generation: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictionStore:
    """Append-only record store with resumable unique sample identities."""

    def __init__(
        self,
        path: str | Path,
        *,
        protocol_id: str,
        checkpoint_id: str,
        experiment_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.protocol_id = protocol_id
        self.checkpoint_id = checkpoint_id
        self.experiment_id = experiment_id
        self.completed: set[str] = set()
        if self.path.exists():
            for record in self:
                if record.protocol_id != protocol_id or record.checkpoint_id != checkpoint_id:
                    raise ValueError(
                        "existing prediction store belongs to a different protocol/checkpoint"
                    )
                if experiment_id is not None and record.experiment_id != experiment_id:
                    raise ValueError("existing prediction store belongs to a different experiment")
                if record.sample_id in self.completed:
                    raise ValueError(f"duplicate sample in prediction store: {record.sample_id}")
                self.completed.add(record.sample_id)

    def __iter__(self) -> Iterator[PredictionRecord]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    value["references"] = tuple(value.get("references", ()))
                    value["image_ids"] = tuple(value.get("image_ids", ()))
                    yield PredictionRecord(**value)
                except (json.JSONDecodeError, TypeError) as error:
                    raise ValueError(f"invalid record at {self.path}:{line_number}") from error

    def append(self, record: PredictionRecord) -> None:
        if record.protocol_id != self.protocol_id or record.checkpoint_id != self.checkpoint_id:
            raise ValueError("record identity does not match store")
        if self.experiment_id is not None and record.experiment_id != self.experiment_id:
            raise ValueError("record experiment does not match store")
        if record.sample_id in self.completed:
            raise ValueError(f"sample already recorded: {record.sample_id}")
        payload = (
            json.dumps(asdict(record), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        self.completed.add(record.sample_id)
