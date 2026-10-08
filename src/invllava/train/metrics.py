from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_text


class JsonlMetricWriter:
    """Append-only canonical metrics; external trackers are optional mirrors."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())


def read_metric_history(path: str | Path) -> list[dict[str, Any]]:
    """Read a canonical metric history with strictly increasing update steps."""

    source = Path(path)
    records: list[dict[str, Any]] = []
    previous = 0
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metric JSON at {source}:{line_number}") from error
            step = record.get("step")
            if not isinstance(step, int) or isinstance(step, bool) or step <= previous:
                raise ValueError(
                    f"metric steps must be strictly increasing positive integers; "
                    f"found {step!r} after {previous} at {source}:{line_number}"
                )
            records.append(record)
            previous = step
    return records


def reconcile_metric_history_for_resume(
    path: str | Path,
    *,
    checkpoint_step: int,
) -> Path | None:
    """Move post-checkpoint metrics aside before an exact replay.

    A process may log updates after its latest durable checkpoint. Replaying
    those updates must not create duplicate steps in the canonical metric file.
    The abandoned suffix is preserved rather than deleted.
    """

    source = Path(path)
    if checkpoint_step < 0:
        raise ValueError("checkpoint_step must be non-negative")
    if not source.exists():
        if checkpoint_step:
            raise FileNotFoundError(
                f"cannot resume step {checkpoint_step} without metric history: {source}"
            )
        return None
    records = read_metric_history(source)
    retained = [record for record in records if int(record["step"]) <= checkpoint_step]
    abandoned = [record for record in records if int(record["step"]) > checkpoint_step]
    if not abandoned:
        return None
    stem = f"metrics.orphaned-after-step-{checkpoint_step:07d}"
    attempt = 1
    while True:
        orphan = source.with_name(f"{stem}-attempt-{attempt}.jsonl")
        if not orphan.exists():
            break
        attempt += 1

    def serialize(rows: list[dict[str, Any]]) -> str:
        return "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        )

    atomic_write_text(orphan, serialize(abandoned))
    atomic_write_text(source, serialize(retained))
    return orphan
