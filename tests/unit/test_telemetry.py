from subprocess import CompletedProcess

import pytest

from invllava.runtime.telemetry import _parse_row, query_nvidia_smi


def test_nvidia_smi_row_parser_handles_numeric_and_unavailable_values() -> None:
    record = _parse_row(
        ["0", "GPU-fixture", "91", "14", "2048", "81920", "N/A", "42"],
        observed_at="2026-08-31T00:00:00+00:00",
        monotonic_ns=17,
    )
    assert record["gpu_index"] == 0
    assert record["gpu_uuid"] == "GPU-fixture"
    assert record["gpu_utilization_percent"] == 91.0
    assert record["memory_total_mib"] == 81920.0
    assert record["power_watts"] is None
    assert record["temperature_c"] == 42.0


def test_nvidia_smi_query_honors_sweep_gpu_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def run(command: list[str], **_: object) -> CompletedProcess[str]:
        captured.extend(command)
        return CompletedProcess(command, 0, "3, GPU-fixture, 90, 10, 1024, 81920, 300, 51\n", "")

    monkeypatch.setenv("INVLLAVA_TELEMETRY_GPU_SELECTOR", "3")
    monkeypatch.setattr("invllava.runtime.telemetry.subprocess.run", run)

    records = query_nvidia_smi()

    assert captured[1:3] == ["--id", "3"]
    assert records[0]["gpu_uuid"] == "GPU-fixture"
