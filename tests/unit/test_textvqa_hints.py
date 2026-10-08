import copy

import pytest

from scripts.audit_textvqa_hints import compare, contains_span, diagnose, hint_text


def fixture():
    rows = [
        {
            "sample_id": "1",
            "prediction": "dakota",
            "checkpoint_id": "c",
            "protocol_id": "p",
            "prompt": "question\nReference OCR token: DAKOTA, DIGITAL\nAnswer briefly.",
            "references": ["dakota"] * 10,
        }
    ]
    score = {
        "checkpoint_id": "c",
        "protocol_id": "p",
        "count": 1,
        "value": 1.0,
        "details": {"per_item": {"1": 1.0}},
    }
    return rows, score


def test_official_rescore_and_availability_preserve_inputs():
    rows, score = fixture()
    before = copy.deepcopy((rows, score))
    item = diagnose(rows, score)["per_item"]["1"]
    assert item["supported_reference_in_hints"] and item["prediction_in_hints"]
    assert not item["longer_prediction_contains_supported_reference"]
    assert (rows, score) == before


def test_wrong_hint_selection_is_recorded_without_score_change():
    rows, score = fixture()
    parent = diagnose(rows, score)
    rows[0]["prediction"] = "digital"
    score["value"] = score["details"]["per_item"]["1"] = 0.0
    group = compare(parent, diagnose(rows, score))["supported_answer_present"]
    assert group["delta_points"] == -100
    assert group["declining_items_with_treatment_hint_span"] == 1


def test_reference_substring_does_not_become_official_credit():
    rows, score = fixture()
    rows[0]["prediction"] = "the brand is dakota"
    score["value"] = score["details"]["per_item"]["1"] = 0.0
    item = diagnose(rows, score)["per_item"]["1"]
    assert item["score"] == 0 and item["longer_prediction_contains_supported_reference"]


@pytest.mark.parametrize(
    "text,phrase,expected", [("cat", "at", False), ("cat dog", "cat dog", True), ("cat", "", False)]
)
def test_span_boundaries(text, phrase, expected):
    assert contains_span(text, phrase) == expected


@pytest.mark.parametrize(
    "prompt",
    ["no hints", "\nReference OCR token: x", "\nReference OCR token: x\nReference OCR token: y\n"],
)
def test_bad_hint_structure_rejected(prompt):
    with pytest.raises(ValueError):
        hint_text(prompt)


def test_stored_score_mismatch_rejected():
    rows, score = fixture()
    score["value"] = 0.0
    with pytest.raises(ValueError, match="rescoring"):
        diagnose(rows, score)


def test_empty_hint_and_no_consensus():
    rows, score = fixture()
    rows[0]["prompt"] = "q\nReference OCR token: \nAnswer."
    rows[0]["references"] = [str(i) for i in range(10)]
    rows[0]["prediction"] = "absent"
    score["value"] = score["details"]["per_item"]["1"] = 0.0
    result = diagnose(rows, score)
    assert not result["per_item"]["1"]["supported_reference_in_hints"]
    assert compare(result, result)["supported_answer_present"]["count"] == 0
