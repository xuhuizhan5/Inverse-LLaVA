import hashlib

import pytest

from invllava.eval.types import EvaluationExample
from scripts.rescore_published_answers import validated_answers


def fixture():
    examples = [EvaluationExample("1", "question", (), ("answer",))]
    records = [
        {
            "sample_id": "1",
            "prediction": "answer",
            "checkpoint_id": "model",
            "inference_protocol_id": "earlier-metadata-id",
            "scoring_protocol_id": "protocol",
            "prompt_sha256": hashlib.sha256(b"question").hexdigest(),
        }
    ]
    return records, examples


def test_published_answers_preserve_answer_text():
    records, examples = fixture()
    assert validated_answers(records, examples, "protocol") == {"1": "answer"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_id", "unknown"),
        ("scoring_protocol_id", "other"),
        ("prompt_sha256", "wrong"),
        ("prediction", 1),
    ],
)
def test_published_answers_reject_mismatches(field, value):
    records, examples = fixture()
    records[0][field] = value
    with pytest.raises(ValueError):
        validated_answers(records, examples, "protocol")


def test_published_answers_reject_duplicates_missing_and_mixed_models():
    records, examples = fixture()
    for invalid in ([], records * 2, [*records, {**records[0], "checkpoint_id": "other"}]):
        with pytest.raises(ValueError):
            validated_answers(invalid, examples, "protocol")
