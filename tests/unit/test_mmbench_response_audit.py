from dataclasses import replace

import pytest

from invllava.eval.types import EvaluationExample
from scripts.audit_mmbench_responses import audit_responses


def examples():
    return [
        EvaluationExample(
            id="1", prompt="q", images=(), choices=("cat", "dog"), references=("A",), group_id="1"
        ),
        EvaluationExample(
            id="1000001",
            prompt="q",
            images=(),
            choices=("dog", "cat"),
            references=("B",),
            group_id="1",
        ),
        EvaluationExample(
            id="2", prompt="q", images=(), choices=("sun", "moon"), references=("A",), group_id="2"
        ),
        EvaluationExample(
            id="1000002",
            prompt="q",
            images=(),
            choices=("moon", "sun"),
            references=("B",),
            group_id="2",
        ),
    ]


def test_unparsed_only_ceiling_preserves_parsed_errors():
    result = audit_responses(examples(), {"1": "A", "1000001": "?", "2": "B", "1000002": "?"})
    assert result["circular_accuracy"] == 0
    assert result["base_rotation_accuracy"] == 0.5
    assert result["all_rotation_accuracy"] == 0.25
    assert result["parsing_only_max_gain_pp"] == 50
    assert result["group_outcomes"] == {"unparsed_only": 1, "has_parsed_error": 1}
    assert result["unparsed_ids"] == ["1000001", "1000002"]


def test_rotation_metric_is_separate_from_base_accuracy():
    result = audit_responses(examples(), {"1": "A", "1000001": "B", "2": "A", "1000002": "A"})
    assert result["circular_accuracy"] == 0.5
    assert result["base_rotation_accuracy"] == 1
    assert result["all_rotation_accuracy"] == 0.75
    assert result["parsing_only_max_gain_pp"] == 0


def test_case_diagnostic_preserves_declared_score_and_wrong_answers():
    predictions = {"1": " a ", "1000001": "b", "2": "b", "1000002": "b"}
    original = dict(predictions)
    result = audit_responses(examples(), predictions)
    case = result["case_only_diagnostic"]
    assert predictions == original
    assert result["circular_accuracy"] == 0
    assert case["circular_accuracy"] == 0.5
    assert case["changed_rotations"] == 4
    assert case["recovered_group_ids"] == ["1"]
    assert case["gain_pp"] == 50
    assert case["remaining_unparsed"] == 0


@pytest.mark.parametrize("answer", ["c", "a.", "a or b", "", "A"])
def test_case_diagnostic_restricts_transformation(answer):
    rows = examples()
    predictions = {row.id: answer for row in rows}
    result = audit_responses(rows, predictions)
    assert result["case_only_diagnostic"]["changed_rotations"] == 0
    assert result["case_only_diagnostic"]["circular_accuracy"] == result["circular_accuracy"]


def test_case_diagnostic_preserves_already_parsed_option_text():
    rows = [replace(row, choices=("cat", "a")) for row in examples()]
    result = audit_responses(rows, {row.id: "a" for row in rows})
    assert result["extracted_answers"] == {"B": 4}
    assert result["case_only_diagnostic"]["changed_rotations"] == 0


def test_case_transformation_does_not_depend_on_references():
    rows = examples()
    predictions = {row.id: "a" for row in rows}
    original = audit_responses(rows, predictions)["case_only_diagnostic"]
    swapped = [replace(row, references=("A",)) for row in rows]
    changed = audit_responses(swapped, predictions)["case_only_diagnostic"]
    assert original["changed_ids"] == changed["changed_ids"]
    assert original["circular_accuracy"] != changed["circular_accuracy"]


@pytest.mark.parametrize("mode", ["missing", "extra", "duplicate"])
def test_invalid_coverage_rejected(mode):
    rows = examples()
    predictions = {row.id: "A" for row in rows}
    if mode == "missing":
        predictions.pop("1")
    elif mode == "extra":
        predictions["unknown"] = "A"
    else:
        rows.append(replace(rows[0]))
    with pytest.raises(ValueError, match="unique, complete"):
        audit_responses(rows, predictions)
