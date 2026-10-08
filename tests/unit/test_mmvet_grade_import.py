"""Receipt import tests use synthetic grades, never benchmark measurements."""

import json
import zipfile

import pytest

from invllava.artifacts.hashing import sha256_file
from invllava.eval.protocols.mmvet import HostedMMVetProtocol
from invllava.eval.types import EvaluationExample


def receipt(tmp_path, mutate_raw=None):
    predictions = {f"v1_{i}": f"Synthetic answer {i}" for i in range(218)}
    examples = [EvaluationExample(key, "Synthetic prompt", ()) for key in predictions]
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps(predictions))
    grading = tmp_path / "grading"
    grading.mkdir()
    request = grading / "request.json"
    request.write_text(
        json.dumps(
            {
                "submission_sha256": sha256_file(submission),
                "space_commit": "ad769f03ae36b7a278bc3c72b912be892b2c4b36",
                "annotations_sha256": (
                    "6eced3121b9865098ea72aca233ca5ff7fd7b81374494d878815b9f856442d8e"
                ),
                "judge": "gpt-4.1",
                "grading_runs": 1,
            }
        )
    )
    grades = {
        key: {"score": [0.5], "content": ["0.5"], "model": ["gpt-4.1-2025-04-14"]}
        for key in predictions
    }
    if mutate_raw:
        mutate_raw(grades)
    archive = grading / "raw.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("synthetic-grade-1runs_gpt-4.1.json", json.dumps(grades))
    accepted = grading / "accepted-grade.json"
    accepted.write_text(
        json.dumps(
            {
                "request_sha256": sha256_file(request),
                "grading_zip_sha256": sha256_file(archive),
                "judge_model": "gpt-4.1-2025-04-14",
                "status": "passed",
                "grading_runs": 1,
                "count": 218,
                "per_item": dict.fromkeys(predictions, 0.5),
                "value": 0.5,
            }
        )
    )
    return HostedMMVetProtocol("fixture", grading, submission), predictions, examples


def test_import_retains_continuous_grades_and_receipt_hashes(tmp_path):
    protocol, predictions, examples = receipt(tmp_path)
    result = protocol.score(predictions, examples)
    assert result.value == 0.5 and result.count == 218
    assert result.details["per_item"] == dict.fromkeys(predictions, 0.5)
    assert result.details["submission_sha256"] == sha256_file(protocol.submission)


@pytest.mark.parametrize(
    "change",
    [
        "answer",
        "submission_bytes",
        "request_bytes",
        "archive_bytes",
        "duplicate_example",
        "missing_prediction",
        "aggregate",
        "nan_aggregate",
        "per_item",
        "judge",
        "missing_archive",
    ],
)
def test_rejects_changed_or_incomplete_receipts(tmp_path, change):
    protocol, predictions, examples = receipt(tmp_path)
    if change == "answer":
        predictions["v1_0"] = "Different answer"
    elif change == "submission_bytes":
        protocol.submission.write_text(protocol.submission.read_text() + "\n")
    elif change == "request_bytes":
        path = protocol.grading_dir / "request.json"
        path.write_text(path.read_text() + "\n")
    elif change == "archive_bytes":
        with (protocol.grading_dir / "raw.zip").open("ab") as handle:
            handle.write(b"modified")
    elif change == "duplicate_example":
        examples.append(examples[0])
    elif change == "missing_prediction":
        predictions.pop("v1_0")
    elif change == "missing_archive":
        (protocol.grading_dir / "raw.zip").rename(protocol.grading_dir / "raw.moved")
    else:
        path = protocol.grading_dir / "accepted-grade.json"
        data = json.loads(path.read_text())
        if change == "aggregate":
            data["value"] = 0.6
        elif change == "nan_aggregate":
            data["value"] = float("nan")
        elif change == "per_item":
            data["per_item"]["v1_0"] = 0.6
        elif change == "judge":
            data["judge_model"] = "gpt-4-0613"
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        protocol.score(predictions, examples)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"score": [0.5]},
        {"score": [True], "content": ["1"], "model": ["gpt-4.1-2025-04-14"]},
        {"score": [0.5], "content": ["error"], "model": ["gpt-4.1-2025-04-14"]},
        {"score": [0.5], "content": ["0.5"], "model": ["different"]},
    ],
)
def test_rejects_invalid_raw_grade_even_with_matching_archive_hash(tmp_path, raw):
    protocol, predictions, examples = receipt(tmp_path, lambda grades: grades.update(v1_0=raw))
    with pytest.raises(ValueError):
        protocol.score(predictions, examples)
