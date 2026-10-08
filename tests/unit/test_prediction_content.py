"""Content parity must retain every declared sample and field."""

import json

import pytest

from scripts.compare_prediction_content import compare_content


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def row(sample_id, prediction="A"):
    return {"sample_id": sample_id, "prediction": prediction}


@pytest.mark.parametrize("policy,passed", [("exact", False), ("right-subset", True)])
def test_subset_is_explicit(tmp_path, policy, passed):
    left = write_rows(tmp_path / "full.jsonl", [row("1"), row("2")])
    right = write_rows(tmp_path / "screen.jsonl", [row("2")])
    result = compare_content(left, right, ("prediction",), id_policy=policy)
    assert result["passed"] is passed
    assert result["compared_count"] == 1
    assert result["missing_right_count"] == 1


def test_subset_requires_every_screen_id(tmp_path):
    left = write_rows(tmp_path / "full.jsonl", [row("1")])
    right = write_rows(tmp_path / "screen.jsonl", [row("1"), row("2")])
    assert not compare_content(left, right, ("prediction",), id_policy="right-subset")["passed"]


def test_response_case_is_not_normalized(tmp_path):
    left = write_rows(tmp_path / "a.jsonl", [row("1", "A")])
    right = write_rows(tmp_path / "b.jsonl", [row("1", "a")])
    result = compare_content(left, right, ("prediction",))
    assert not result["passed"] and result["mismatch_count"] == 1


@pytest.mark.parametrize("missing", ["left", "right", "both"])
def test_absent_fields_cannot_pass(tmp_path, missing):
    left_row, right_row = row("1"), row("1")
    if missing in ("left", "both"):
        left_row.pop("prediction")
    if missing in ("right", "both"):
        right_row.pop("prediction")
    left = write_rows(tmp_path / "a.jsonl", [left_row])
    right = write_rows(tmp_path / "b.jsonl", [right_row])
    result = compare_content(left, right, ("prediction",))
    assert not result["passed"]
    assert result["mismatch_examples"][0]["missing_in"]


def test_order_does_not_change_parity(tmp_path):
    left = write_rows(tmp_path / "a.jsonl", [row("1"), row("2", "B")])
    right = write_rows(tmp_path / "b.jsonl", [row("2", "B"), row("1")])
    assert compare_content(left, right, ("prediction",))["passed"]


@pytest.mark.parametrize("rows", [[], [row("1"), row("1")]])
def test_invalid_input_rejected(tmp_path, rows):
    left = write_rows(tmp_path / "a.jsonl", rows)
    right = write_rows(tmp_path / "b.jsonl", [row("1")])
    with pytest.raises(ValueError):
        compare_content(left, right, ("prediction",))


@pytest.mark.parametrize("fields", [(), ("prediction", "prediction")])
def test_invalid_field_selection_rejected(tmp_path, fields):
    with pytest.raises(ValueError):
        compare_content(tmp_path / "a", tmp_path / "b", fields)


def test_invalid_policy_rejected(tmp_path):
    with pytest.raises(ValueError, match="sample-ID"):
        compare_content(tmp_path / "a", tmp_path / "b", id_policy="intersection")
