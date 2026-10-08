import json
from pathlib import Path

import pytest

from invllava.eval.language_compare import compare_language_results


def _write_result(path: Path, values: list[float], *, prompt_suffix: str = "") -> None:
    samples = []
    for index, value in enumerate(values):
        samples.append(
            {
                "doc_hash": f"doc-{index}",
                "prompt_hash": f"prompt-{index}{prompt_suffix}",
                "target_hash": f"target-{index}",
                "filter": "none",
                "acc_norm": value,
            }
        )
    path.write_text(json.dumps({"samples": {"fixture": samples}}), encoding="utf-8")


def test_compare_language_results_validates_and_bootstraps_pairs(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_result(left, [1.0, 1.0, 0.0])
    _write_result(right, [0.0, 0.0, 0.0])

    result = compare_language_results(
        left, right, metric="acc_norm", filter_name="none", resamples=100, seed=7
    )

    assert result["sample_count"] == 3
    assert result["left_mean"] == pytest.approx(2 / 3)
    assert result["right_mean"] == 0.0
    assert result["estimate"] == pytest.approx(2 / 3)


def test_compare_language_results_refuses_prompt_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_result(left, [1.0])
    _write_result(right, [1.0], prompt_suffix="-changed")

    with pytest.raises(ValueError, match="different prompts or targets"):
        compare_language_results(left, right, metric="acc_norm", filter_name="none")
