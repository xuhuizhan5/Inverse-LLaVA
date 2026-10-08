from pathlib import Path

from invllava.eval.types import EvaluationExample
from invllava.prompting import format_vicuna_v1_user_prompt
from scripts.verify_scienceqa_golden import SINGLE_PRED_SUFFIX, compare_official_inputs


def test_scienceqa_audit_accepts_released_prompt_and_ignores_text_only_questions():
    questions = [
        {"id": "4", "conversations": []},
        {
            "id": "5",
            "image": "5/image.png",
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion?\nA. one\nB. two"},
                {"from": "gpt", "value": "B"},
            ],
        },
    ]
    example = EvaluationExample(
        id="5",
        images=(Path("image.png"),),
        references=("B",),
        choices=("one", "two"),
        prompt=format_vicuna_v1_user_prompt(
            questions[1]["conversations"][0]["value"] + SINGLE_PRED_SUFFIX
        ),
    )
    assert compare_official_inputs(questions, [example])["passed"]


def test_scienceqa_audit_rejects_same_count_with_different_ids_or_prompt():
    questions = [
        {
            "id": "5",
            "image": "5/image.png",
            "conversations": [
                {"from": "human", "value": "<image>\nQuestion?\nA. one\nB. two"},
                {"from": "gpt", "value": "B"},
            ],
        }
    ]
    wrong_id = EvaluationExample(
        id="1",
        prompt="alternative prompt",
        images=(Path("image.png"),),
        references=("B",),
        choices=("one", "two"),
    )
    result = compare_official_inputs(questions, [wrong_id])
    assert not result["passed"]
    assert result["missing_ids"] == ["5"]
    assert result["extra_ids"] == ["1"]
    wrong_prompt = EvaluationExample(
        id="5",
        prompt="alternative prompt",
        images=(Path("image.png"),),
        references=("B",),
        choices=("one", "two"),
    )
    assert compare_official_inputs(questions, [wrong_prompt])["mismatches"] == [
        {"sample_id": "5", "fields": ["prompt"]}
    ]
