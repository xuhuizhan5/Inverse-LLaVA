from pathlib import Path

import pytest

from invllava.eval.protocols.mme import MMEProtocol
from invllava.eval.protocols.multiple_choice import MultipleChoiceProtocol
from invllava.eval.protocols.scienceqa import (
    ScienceQAProtocol,
    extract_llava_scienceqa_choice,
)
from invllava.eval.types import EvaluationExample


def test_multiple_choice_extracts_letter_and_text() -> None:
    examples = [
        EvaluationExample("1", "", (), ("B",), ("red", "blue")),
        EvaluationExample("2", "", (), ("A",), ("one", "two")),
    ]
    score = MultipleChoiceProtocol("fixture").score({"1": "The answer is B.", "2": "one"}, examples)
    assert score.value == 1.0


def test_scienceqa_extraction_matches_llava_rules() -> None:
    assert extract_llava_scienceqa_choice("B", 3) == "B"
    assert extract_llava_scienceqa_choice("B. because", 3) == "B"
    assert extract_llava_scienceqa_choice("The answer is B.", 3) == "B"
    assert extract_llava_scienceqa_choice("I choose B", 3) is None
    assert extract_llava_scienceqa_choice("blue", 3) is None

    example = EvaluationExample("1", "", (), ("B",), ("red", "blue", "green"))
    score = ScienceQAProtocol("fixture").score({"1": "The answer is B."}, [example])
    assert score.value == 1.0


@pytest.mark.parametrize(
    "response,count,expected",
    [
        (" B ", 3, None),
        ("B\n", 3, None),
        ("B. ", 3, "B"),
        ("C. The answer is B.", 2, None),
        ("F. The answer is B.", 2, "B"),
        ("B. The answer is A.", 2, "B"),
        ("The answer is B. The answer is A.", 2, None),
        ("The answer is C.", 2, None),
        ("b", 3, None),
    ],
)
def test_scienceqa_official_raw_boundary(response, count, expected):
    assert extract_llava_scienceqa_choice(response, count) == expected


@pytest.mark.parametrize("count", [0, 6, -1, True, 2.0])
def test_scienceqa_invalid_choice_count(count):
    with pytest.raises(ValueError, match="one to five"):
        extract_llava_scienceqa_choice("A", count)


def test_mme_accuracy_plus_groups_questions_by_image() -> None:
    examples = [
        EvaluationExample(
            str(index),
            "",
            (Path("x"),),
            (answer,),
            group_id=image,
            metadata={"category": "existence"},
        )
        for index, (image, answer) in enumerate(
            (("a", "yes"), ("a", "no"), ("b", "yes"), ("b", "no"))
        )
    ]
    examples = [
        EvaluationExample(
            item.id,
            item.prompt,
            item.images,
            item.references,
            item.choices,
            item.group_id,
            {"category": "existence", "domain": "perception"},
        )
        for item in examples
    ]
    score = MMEProtocol("fixture", domain="perception").score(
        {"0": "yes", "1": "no", "2": "yes", "3": "yes"}, examples
    )
    assert score.value == 125.0  # 75 accuracy + 50 accuracy-plus
