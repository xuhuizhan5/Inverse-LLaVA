import json
from pathlib import Path

import pytest

from scripts.audit_mme_official import grouped_rows


def _row(index: int, answer: str = "Yes") -> dict:
    return {
        "sample_id": f"OCR/a.png/{index}",
        "checkpoint_id": "model-1",
        "metadata": {"category": "OCR"},
        "references": ["Yes" if index == 0 else "No"],
        "prediction": answer,
        "prompt": "Question",
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_official_audit_restores_pair_order(tmp_path: Path) -> None:
    source = _write(tmp_path / "predictions.jsonl", [_row(1), _row(0)])
    assert [r["sample_id"] for r in grouped_rows([source])["OCR"]] == [
        "OCR/a.png/0",
        "OCR/a.png/1",
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [_row(0)],
        [_row(0), _row(0)],
        [_row(0), _row(1, "No\nBecause")],
        [_row(0), {**_row(1), "checkpoint_id": "model-2"}],
    ],
)
def test_official_audit_rejects_ambiguous_input(tmp_path: Path, rows: list[dict]) -> None:
    with pytest.raises(ValueError):
        grouped_rows([_write(tmp_path / "predictions.jsonl", rows)])
