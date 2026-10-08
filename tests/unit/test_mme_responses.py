import pytest

from invllava.eval.types import EvaluationExample
from scripts.audit_mme_responses import audit_responses, category_comparisons


def examples():
    return [
        EvaluationExample(
            id=str(index),
            prompt="question",
            images=(),
            references=(answer,),
            group_id=str(index // 2),
            metadata={"category": "test", "domain": "cognition"},
        )
        for index, answer in enumerate(("yes", "no", "yes", "no"))
    ]


def test_constant_no_score_does_not_imply_paired_success():
    result = audit_responses(examples(), dict.fromkeys(("0", "1", "2", "3"), "No."))
    assert result["score"] == 50
    assert result["constant_answer_controls"] == {"yes": 50, "no": 50}
    assert result["categories"]["test"]["both_correct_pairs"] == 0
    assert result["categories"]["test"]["answers"] == {"no": 4}


def test_response_audit_counts_invalid_and_both_correct_pairs():
    result = audit_responses(examples(), {"0": "Yes", "1": "No", "2": "I think yes", "3": "n"})
    assert result["score"] == 125
    assert result["categories"]["test"]["both_correct_pairs"] == 1
    assert result["categories"]["test"]["answers"]["invalid"] == 1


def test_response_audit_rejects_missing_predictions():
    with pytest.raises(ValueError, match="coverage"):
        audit_responses(examples(), {"0": "yes"})


def scores():
    identity = {
        "benchmark": "mme-cognition",
        "examples_sha256": "synthetic-input",
        "protocol_id": "synthetic-protocol",
        "scorer_id": "synthetic-scorer",
    }
    return (
        {**identity, "value": 200, "details": {"per_item": dict.fromkeys("0123", 1)}},
        {
            **identity,
            "value": 50,
            "details": {"per_item": dict(zip("0123", [0, 1, 0, 1], strict=True))},
        },
    )


def test_category_comparison_uses_native_accuracy_plus():
    result = category_comparisons(examples(), *scores(), resamples=100)["test"]
    assert result["estimate"] == result["low"] == result["high"] == 150
    assert result["questions"] == 4 and result["image_groups"] == 2
    assert result["unit"] == "mme_image_group"


@pytest.mark.parametrize("field", ["benchmark", "examples_sha256", "protocol_id", "scorer_id"])
def test_category_comparison_rejects_identity_mismatch(field):
    left, right = scores()
    right[field] = "different"
    with pytest.raises(ValueError, match="incompatible"):
        category_comparisons(examples(), left, right, resamples=10)


def test_category_comparison_rejects_nonbinary_values():
    left, right = scores()
    left["details"]["per_item"]["0"] = float("nan")
    with pytest.raises(ValueError, match="binary"):
        category_comparisons(examples(), left, right, resamples=10)


def test_category_comparison_rejects_aggregate_drift():
    left, right = scores()
    left["value"] = 199
    with pytest.raises(ValueError, match="aggregate"):
        category_comparisons(examples(), left, right, resamples=10)
