"""Low-frequency, provider-neutral NVIDIA telemetry for paid training runs."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_FIELDS = (
    "index",
    "uuid",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)


def _parse_row(row: list[str], *, observed_at: str, monotonic_ns: int) -> dict[str, Any]:
    if len(row) != len(_FIELDS):
        raise ValueError(f"nvidia-smi returned {len(row)} columns, expected {len(_FIELDS)}")
    values = [value.strip() for value in row]
    numeric = [int(values[0])]
    for value in values[2:]:
        if value in {"", "N/A", "[N/A]"}:
            numeric.append(None)
        else:
            numeric.append(float(value))
    return {
        "observed_at": observed_at,
        "monotonic_ns": monotonic_ns,
        "gpu_index": numeric[0],
        "gpu_uuid": values[1],
        "gpu_utilization_percent": numeric[1],
        "memory_utilization_percent": numeric[2],
        "memory_used_mib": numeric[3],
        "memory_total_mib": numeric[4],
        "power_watts": numeric[5],
        "temperature_c": numeric[6],
    }


def query_nvidia_smi() -> list[dict[str, Any]]:
    observed_at = datetime.now(timezone.utc).isoformat()
    monotonic_ns = time.monotonic_ns()
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    selector = os.environ.get("INVLLAVA_TELEMETRY_GPU_SELECTOR", "").strip()
    if selector:
        command[1:1] = ["--id", selector]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5)
    rows = list(csv.reader(line for line in completed.stdout.splitlines() if line.strip()))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return [_parse_row(row, observed_at=observed_at, monotonic_ns=monotonic_ns) for row in rows]


class NvidiaSmiMonitor:
    """Poll ``nvidia-smi`` in one daemon thread and append fsynced JSONL records."""

    def __init__(self, path: str | Path, *, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="invllava-nvidia-smi", daemon=True)

    def start(self) -> None:
        # Fail before paid training if telemetry was requested but the executable
        # or its query contract is unavailable. Subsequent transient failures are
        # recorded without terminating the scientific run.
        first = query_nvidia_smi()
        self._append(first)
        self._thread.start()

    def _append(self, records: list[dict[str, Any]]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._append(query_nvidia_smi())
            except Exception as error:  # Diagnostic failure must not kill training.
                self._append(
                    [
                        {
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "monotonic_ns": time.monotonic_ns(),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    ]
                )

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds + 1.0))
        if self._thread.is_alive():
            raise RuntimeError("hardware telemetry thread did not stop")
