import numpy as np
import pytest

from invllava.analysis.figures import save_grouped_score_breakdown, save_profile_comparison
from invllava.analysis.representations import (
    cka_sample_uncertainty,
    effective_rank,
    linear_cka,
    matched_permutation_test,
    matched_shuffled_margin,
    paired_numerical_similarity,
    select_layer_views,
    singular_spectrum,
)


def test_neighbor_overlap_uses_the_closest_other_sample():
    from invllava.analysis.representations import knn_overlap

    def points(degrees):
        radians = np.deg2rad(degrees)
        return np.column_stack((np.cos(radians), np.sin(radians)))

    left = points([0, 10, 40, 110])
    right = points([0, 80, 90, 180])
    assert knn_overlap(left, right, k=1) == 0.75
    assert knn_overlap(left, right, k=3) == 1.0


def test_linear_cka_identical_is_one() -> None:
    rng = np.random.default_rng(7)
    value = rng.normal(size=(50, 8))
    assert np.isclose(linear_cka(value, value), 1.0)


def test_unbiased_hsic_cka_preserves_scalar_translation_and_rotation():
    rng = np.random.default_rng(2026)
    values = rng.normal(size=(30, 8))
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    assert np.isclose(linear_cka(values, 3 * values @ rotation + 7, debiased=True), 1)


def test_unbiased_hsic_cka_retains_negative_estimates():
    left = np.array([[0], [0], [1], [1]])
    right = np.array([[0], [1], [0], [1]])
    assert np.isclose(linear_cka(left, right, debiased=True), -0.5)


def test_unbiased_hsic_cka_reduces_high_dimensional_independence_bias():
    rng = np.random.default_rng(2026)
    left, right = rng.normal(size=(100, 4096)), rng.normal(size=(100, 4096))
    assert linear_cka(left, right) > 0.95
    assert abs(linear_cka(left, right, debiased=True)) < 0.08


@pytest.mark.parametrize("value", [np.ones((4, 3)), np.eye(4)])
def test_unbiased_hsic_cka_rejects_degenerate_geometry(value):
    with pytest.raises(ValueError, match="undefined"):
        linear_cka(value, value, debiased=True)


def test_unbiased_hsic_cka_requires_four_samples():
    with pytest.raises(ValueError, match="four"):
        linear_cka(np.eye(3), np.eye(3), debiased=True)


def test_cka_uncertainty_matches_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(13)
    left, right = rng.normal(size=(20, 24)), rng.normal(size=(20, 9))
    result = cka_sample_uncertainty(left, right, resamples=99)
    assert np.isclose(result["cka"], linear_cka(left, right))
    assert result == cka_sample_uncertainty(left, right, resamples=99)
    assert result["ci_low"] <= result["ci_high"]
    assert 0 < result["one_sided_p_value"] <= 1


def test_cka_uncertainty_preserves_affine_scalar_equivalence():
    value = np.random.default_rng(11).normal(size=(20, 8))
    result = cka_sample_uncertainty(value, value * 5 + 10, resamples=99)
    assert np.isclose(result["ci_low"], 1)
    assert np.isclose(result["ci_high"], 1)
    assert result["one_sided_p_value"] == 0.01


def test_cka_uncertainty_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="aligned"):
        cka_sample_uncertainty(np.eye(4), np.eye(5))
    with pytest.raises(ValueError, match="resamples"):
        cka_sample_uncertainty(np.eye(4), np.eye(4), resamples=0)
    with pytest.raises(ValueError, match="undefined"):
        cka_sample_uncertainty(np.ones((4, 5)), np.eye(4))


def test_linear_cka_marks_constant_representations_undefined() -> None:
    with pytest.raises(ValueError, match="undefined"):
        linear_cka(np.ones((4, 8)), np.arange(32).reshape(4, 8))


def test_layer_selection_separates_pooling_and_records_constant_layers() -> None:
    variable = np.arange(32).reshape(4, 8)
    arrays = {
        "hidden.0": variable,
        "hidden.1": variable,
        "hidden.last.0": np.ones((4, 8)),
        "hidden.last.1": variable,
        "hidden.last.10": variable,
        "hidden.last.2": variable,
    }
    assert select_layer_views(arrays, "hidden.") == (["hidden.0", "hidden.1"], [])
    assert select_layer_views(arrays, "hidden.last.") == (
        ["hidden.last.1", "hidden.last.2", "hidden.last.10"],
        ["hidden.last.0"],
    )
    with pytest.raises(ValueError, match="no nonconstant"):
        select_layer_views({"hidden.0": np.ones((4, 8))}, "hidden.")


def test_linear_cka_sample_gram_matches_feature_product() -> None:
    rng = np.random.default_rng(11)
    left = rng.normal(size=(12, 64))
    right = rng.normal(size=(12, 48))
    centered_left = left - left.mean(axis=0, keepdims=True)
    centered_right = right - right.mean(axis=0, keepdims=True)
    # Some Accelerate-linked NumPy builds raise stale BLAS floating-point flags
    # for these finite products. The implementation validates finite outputs.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        expected = np.linalg.norm(centered_left.T @ centered_right, ord="fro") ** 2 / (
            np.linalg.norm(centered_left.T @ centered_left, ord="fro")
            * np.linalg.norm(centered_right.T @ centered_right, ord="fro")
        )
    assert np.isclose(linear_cka(left, right), expected)


def test_effective_rank_and_margin_are_finite() -> None:
    value = np.eye(12)
    assert effective_rank(value)["participation_ratio"] > 1
    assert matched_shuffled_margin(value, value, seed=2) > 0
    assert np.isclose(sum(singular_spectrum(value)), 1.0)


def test_matched_permutation_test_detects_exact_pairing() -> None:
    value = np.eye(24)
    result = matched_permutation_test(value, value, permutations=999, seed=7)
    assert result["matched_cosine_mean"] == 1.0
    assert result["matched_minus_null_mean"] > 0.9
    assert result["one_sided_p_value"] <= 0.002
    assert result["permutations"] == 999


def test_matched_permutation_test_validates_contract() -> None:
    value = np.eye(4)
    with pytest.raises(ValueError, match="permutations"):
        matched_permutation_test(value, value, permutations=0)
    with pytest.raises(ValueError, match="confidence"):
        matched_permutation_test(value, value, confidence=1.0)


def test_paired_numerical_similarity_reports_drift() -> None:
    reference = np.eye(4)
    result = paired_numerical_similarity(reference + 0.01, reference)
    assert result["mean_paired_cosine"] < 1.0
    assert np.isclose(result["rmse"], 0.01)
    assert result["relative_frobenius_error"] > 0
    assert np.isclose(result["max_absolute_error"], 0.01)
    with pytest.raises(ValueError, match="identical shapes"):
        paired_numerical_similarity(reference, reference[:, :2])


def test_linear_cka_rescales_large_values_and_rejects_nan() -> None:
    value = np.arange(40, dtype=float).reshape(10, 4) * 1e200
    assert np.isclose(linear_cka(value, value), 1.0)
    value[0, 0] = np.nan
    try:
        linear_cka(value, value)
    except ValueError as error:
        assert "NaN" in str(error)
    else:
        raise AssertionError("NaN input should fail")


def test_score_breakdown_rejects_misaligned_categories() -> None:
    try:
        save_grouped_score_breakdown(
            {"left": {"a": 1.0}, "right": {"b": 1.0}},
            "unused.pdf",
            title="fixture",
            ylabel="score",
        )
    except ValueError as error:
        assert "identical ordered categories" in str(error)
    else:
        raise AssertionError("misaligned breakdown categories must fail")


def test_profile_comparison_requires_matching_runtime_contract(tmp_path) -> None:
    def profile(hardware: str) -> dict:
        return {
            "hardware": hardware,
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "fixed_decode_policy": "fixed",
            "total_parameters": 10,
            "trainable_parameters": 2,
            "profiles": {
                "1": {
                    "autoregressive": {
                        "median_decode_tokens_per_second": 4.0,
                        "median_time_to_first_token_seconds": 0.1,
                        "peak_allocated_bytes": 1024**3,
                    }
                }
            },
        }

    with pytest.raises(ValueError, match="one hardware and runtime contract"):
        save_profile_comparison(
            {"left": profile("gpu-a"), "right": profile("gpu-b")}, tmp_path / "plot.pdf"
        )


def test_profile_comparison_keeps_training_and_runtime_parameter_semantics(tmp_path) -> None:
    def profile(*, training: int | None, runtime: int | None) -> dict:
        return {
            "hardware": "gpu",
            "dtype": "bfloat16",
            "attention_backend": "sdpa",
            "fixed_decode_policy": "fixed",
            "total_parameters": 10,
            "trainable_parameters": training,
            "runtime_requires_grad_parameters": runtime,
            "profiles": {
                "1": {
                    "autoregressive": {
                        "median_decode_tokens_per_second": 4.0,
                        "median_time_to_first_token_seconds": 0.1,
                        "peak_allocated_bytes": 1024**3,
                    }
                }
            },
        }

    metadata = save_profile_comparison(
        {
            "native": profile(training=2, runtime=None),
            "reference": profile(training=None, runtime=10),
        },
        tmp_path / "plot.pdf",
    )

    assert metadata["parameter_counts"]["native"]["training_trainable"] == 2
    assert metadata["parameter_counts"]["reference"]["training_trainable"] is None
    assert metadata["parameter_counts"]["reference"]["runtime_requires_grad"] == 10
    assert metadata["series"]["native"]["1"]["median_time_to_first_token_seconds"] == 0.1
    assert metadata["series"]["native"]["1"]["peak_allocated_bytes"] == 1024**3
    assert metadata["series_units"]["peak_allocated_bytes"] == "bytes"
    assert b"/Subtype /Type3" not in (tmp_path / "plot.pdf").read_bytes()
