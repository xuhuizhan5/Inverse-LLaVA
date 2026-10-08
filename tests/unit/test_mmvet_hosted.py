import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("mmvet_hosted", Path("scripts/mmvet_hosted.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example():
    return {"v1_0": {"score": [0.5], "content": ["0.5"], "model": ["gpt-4.1-2025-04-14"]}}


def test_hosted_grade_preserves_continuous_score():
    assert MODULE.checked_grades(example(), {"v1_0"})["value"] == 0.5


@pytest.mark.parametrize(
    "key,value",
    [
        ("score", [float("nan")]),
        ("score", [True]),
        ("score", [0]),
        ("score", []),
        ("content", ["error"]),
        ("model", ["gpt-4-0613"]),
    ],
)
def test_invalid_or_inconsistent_grade_is_not_a_model_error(key, value):
    records = example()
    records["v1_0"][key] = value
    with pytest.raises(ValueError):
        MODULE.checked_grades(records, {"v1_0"})


def test_missing_grade_rejected():
    with pytest.raises(ValueError):
        MODULE.checked_grades(example(), {"v1_0", "v1_1"})


@pytest.mark.parametrize("record", [None, [], {"score": None}, {"score": 0.5}])
def test_malformed_grade_record_rejected(record):
    with pytest.raises(ValueError):
        MODULE.checked_grades({"v1_0": record}, {"v1_0"})


def test_mixed_judge_versions_rejected():
    records = example()
    records["v1_1"] = copy.deepcopy(records["v1_0"])
    records["v1_1"]["model"] = ["gpt-4.1-other"]
    with pytest.raises(ValueError):
        MODULE.checked_grades(records, {"v1_0", "v1_1"})
