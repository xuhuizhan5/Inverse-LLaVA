import copy

import pytest

from scripts.audit_prediction_responses import audit


def fixture():
    records = [
        {
            "sample_id": str(index),
            "prediction": response,
            "checkpoint_id": "checkpoint",
            "protocol_id": "protocol",
            "metadata": {"question_type": "reading"},
        }
        for index, response in enumerate((" Hello  world ", "hello world", ""))
    ]
    score = {
        "count": 3,
        "checkpoint_id": "checkpoint",
        "protocol_id": "protocol",
        "value": 1 / 3,
        "details": {"per_item": {"0": 1, "1": 0, "2": 0}},
    }
    return records, score


def test_response_summary_is_descriptive_and_preserves_inputs():
    records, score = fixture()
    before = copy.deepcopy((records, score))
    result = audit(records, score, group_field="question_type")
    assert result["overall"]["empty_responses"] == 1
    assert result["overall"]["median_whitespace_words"] == 2
    assert result["overall"]["unique_normalized_responses"] == 2
    assert result["overall"]["most_common_normalized_responses"][0] == {
        "response": "hello world",
        "count": 2,
    }
    assert result["groups"]["reading"]["count"] == 3
    assert result["score"] == 1 / 3
    assert (records, score) == before


@pytest.mark.parametrize(
    "problem", ["duplicate", "missing", "identity", "ids", "nonfinite", "text"]
)
def test_invalid_prediction_inputs_rejected(problem):
    records, score = fixture()
    if problem == "duplicate":
        records[1]["sample_id"] = "0"
    elif problem == "missing":
        records.pop()
    elif problem == "identity":
        records[1]["checkpoint_id"] = "different"
    elif problem == "ids":
        score["details"]["per_item"] = {"x": 0, "y": 0, "z": 0}
    elif problem == "nonfinite":
        score["details"]["per_item"]["0"] = float("nan")
    else:
        records[0]["prediction"] = None
    with pytest.raises(ValueError):
        audit(records, score)


def test_declared_group_field_required_for_every_record():
    records, score = fixture()
    records[0]["metadata"] = {}
    with pytest.raises(ValueError, match="metadata"):
        audit(records, score, group_field="question_type")


def test_no_groups_requested():
    records, score = fixture()
    assert audit(records, score)["groups"] == {}
