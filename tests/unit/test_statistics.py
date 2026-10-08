import numpy as np
import pytest

from invllava.analysis.statistics import (
    grouped_paired_bootstrap,
    mme_paired_bootstrap,
    paired_bootstrap,
    stratified_macro_paired_bootstrap,
)
from invllava.analysis.strata import numeric_score_strata


def test_paired_bootstrap_direction() -> None:
    interval = paired_bootstrap(np.ones(20), np.zeros(20), resamples=100, seed=1)
    assert interval.estimate == 1.0
    assert interval.low == 1.0
    assert interval.high == 1.0


def test_grouped_bootstrap_marks_unit() -> None:
    interval = grouped_paired_bootstrap(
        np.array([1, 1, 0, 0]),
        np.zeros(4),
        np.array(["a", "a", "b", "b"]),
        resamples=100,
    )
    assert interval.unit == "group"


def test_grouped_bootstrap_preserves_item_mean_with_unequal_groups() -> None:
    interval = grouped_paired_bootstrap(
        np.array([1, 1, 1, 0]),
        np.zeros(4),
        np.array(["a", "a", "a", "b"]),
        resamples=200,
        seed=7,
    )
    assert interval.estimate == 0.75
    assert interval.low <= interval.estimate <= interval.high


def test_grouped_bootstrap_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        grouped_paired_bootstrap(np.array([np.nan]), np.zeros(1), np.array(["a"]))


def test_mme_bootstrap_recomputes_accuracy_plus() -> None:
    interval = mme_paired_bootstrap(
        np.ones(4),
        np.array([1, 0, 1, 0]),
        np.array(["a", "a", "b", "b"]),
        np.array(["existence"] * 4),
        resamples=100,
        seed=3,
    )
    assert interval.estimate == 150.0
    assert interval.unit == "mme_image_group"


def test_mme_bootstrap_preserves_unequal_group_weights() -> None:
    interval = mme_paired_bootstrap(
        np.array([1, 1, 1, 0, 0]),
        np.zeros(5),
        np.array(["a", "a", "a", "b", "b"]),
        np.array(["count"] * 5),
        resamples=200,
        seed=7,
    )

    # Accuracy is 3/5 and accuracy-plus is 1/2.
    assert np.isclose(interval.estimate, 110.0)
    assert interval.low <= interval.estimate <= interval.high


def test_stratified_bootstrap_targets_equal_weight_macro_average() -> None:
    interval = stratified_macro_paired_bootstrap(
        np.array([1, 1, 1, 0]),
        np.zeros(4),
        np.array(["large", "large", "large", "small"]),
        resamples=100,
        seed=3,
    )

    assert interval.estimate == 0.5
    assert interval.unit == "fixed_strata_item"


def test_numeric_score_strata_reports_paired_bucket_intervals() -> None:
    result = numeric_score_strata(
        {
            "inverse": {"a": 1.0, "b": 0.0, "c": 1.0},
            "baseline": {"a": 0.0, "b": 0.0, "c": 0.0},
        },
        {
            "a": {"ocr_tokens": 0},
            "b": {"ocr_tokens": 4},
            "c": {"ocr_tokens": 12},
        },
        metadata_key="ocr_tokens",
        boundaries=(0.0, 1.0, 6.0, 11.0),
        primary_model="inverse",
        resamples=100,
        seed=3,
    )

    assert result["sample_count"] == 3
    assert [item["label"] for item in result["strata"]] == ["0", "1-5", "11+"]
    assert result["strata"][0]["paired_intervals"]["inverse-minus-baseline"]["estimate"] == 1.0


def test_numeric_strata_grouped_interval_preserves_item_weighting() -> None:
    result = numeric_score_strata(
        {"left": {"a": 1.0, "b": 1.0, "c": 0.0}, "right": dict.fromkeys("abc", 0.0)},
        {key: {"count": 1} for key in "abc"},
        metadata_key="count",
        boundaries=(0.0,),
        primary_model="left",
        groups_by_id={"a": "image1", "b": "image1", "c": "image2"},
        resamples=200,
        seed=7,
    )
    row = result["strata"][0]
    assert row["group_count"] == 2
    assert result["resampling_unit"] == "image_group"
    interval = row["paired_intervals"]["left-minus-right"]
    assert interval["unit"] == "group"
    assert interval["estimate"] == pytest.approx(2 / 3)
    expected = grouped_paired_bootstrap(
        np.asarray([1.0, 1.0, 0.0]),
        np.zeros(3),
        np.asarray(["image1", "image1", "image2"]),
        resamples=200,
        seed=7,
    )
    assert interval["low"] == expected.low and interval["high"] == expected.high


@pytest.mark.parametrize("groups", [{}, {"a": ""}, {"a": None}, {"a": "x", "b": "y"}])
def test_numeric_strata_rejects_invalid_group_mapping(groups) -> None:
    with pytest.raises(ValueError, match="image groups"):
        numeric_score_strata(
            {"left": {"a": 1.0}, "right": {"a": 0.0}},
            {"a": {"count": 1}},
            metadata_key="count",
            boundaries=(0.0,),
            primary_model="left",
            groups_by_id=groups,
        )


def test_numeric_strata_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        numeric_score_strata(
            {"left": {"a": float("nan")}, "right": {"a": 0.0}},
            {"a": {"count": 1}},
            metadata_key="count",
            boundaries=(0.0,),
            primary_model="left",
        )
