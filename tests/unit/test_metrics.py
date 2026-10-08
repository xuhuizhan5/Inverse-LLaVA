from pathlib import Path
from tempfile import TemporaryDirectory

from invllava.train.metrics import (
    JsonlMetricWriter,
    read_metric_history,
    reconcile_metric_history_for_resume,
)


def test_resume_preserves_abandoned_metric_tail() -> None:
    with TemporaryDirectory() as directory:
        metrics = Path(directory) / "metrics.jsonl"
        writer = JsonlMetricWriter(metrics)
        for step in (1, 2, 3):
            writer.write({"step": step, "train/loss": 4.0 - step})

        orphan = reconcile_metric_history_for_resume(metrics, checkpoint_step=2)

        assert orphan is not None
        assert [record["step"] for record in read_metric_history(metrics)] == [1, 2]
        assert [record["step"] for record in read_metric_history(orphan)] == [3]


def test_metric_history_rejects_duplicate_steps() -> None:
    with TemporaryDirectory() as directory:
        metrics = Path(directory) / "metrics.jsonl"
        metrics.write_text('{"step": 1}\n{"step": 1}\n', encoding="utf-8")
        try:
            read_metric_history(metrics)
        except ValueError as error:
            assert "strictly increasing" in str(error)
        else:
            raise AssertionError("duplicate metric steps were accepted")
