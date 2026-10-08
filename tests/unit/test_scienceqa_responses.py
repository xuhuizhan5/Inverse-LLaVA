import copy

import pytest

from invllava.eval.types import EvaluationExample
from scripts.audit_scienceqa_responses import diagnose


def fixture(prediction="b", reference="B", official=0, invalid=1):
    rows = [
        {
            "sample_id": "1",
            "prompt": "q",
            "references": [reference],
            "prediction": prediction,
            "checkpoint_id": "c",
            "protocol_id": "p",
        }
    ]
    score = {
        "count": 1,
        "value": official,
        "checkpoint_id": "c",
        "protocol_id": "p",
        "details": {"per_item": {"1": official}, "invalid": invalid},
    }
    examples = [EvaluationExample("1", "q", (), (reference,), ("option1", "option2"))]
    return rows, score, examples


def test_case_diagnostic_preserves_official_outputs():
    args = fixture()
    before = copy.deepcopy(args)
    result = diagnose(*args)
    assert result["official_accuracy"] == 0
    assert result["case_only_diagnostic_accuracy"] == 1
    assert result["counts"]["standalone_lowercase_choices"] == 1
    assert args == before


@pytest.mark.parametrize("prediction", ["b.", "option b", "b or a", "z", "", "β", " b ", "b\n"])
def test_diagnostic_does_not_repair_other_formats(prediction):
    result = diagnose(*fixture(prediction))
    assert result["case_only_diagnostic_accuracy"] == 0
    assert result["counts"]["standalone_lowercase_choices"] == 0


def test_uppercase_official_correct_stays_correct():
    result = diagnose(*fixture("B", official=1, invalid=0))
    assert result["official_accuracy"] == result["case_only_diagnostic_accuracy"] == 1


def test_official_score_mismatch_fails():
    with pytest.raises(ValueError, match="per-item"):
        diagnose(*fixture("b", official=1))
