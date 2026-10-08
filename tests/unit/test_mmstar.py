from invllava.eval.protocols.mmstar import MMStarProtocol, extract_mmstar_choice
from invllava.eval.types import EvaluationExample


def _example(sample_id: str, answer: str, category: str, l2: str) -> EvaluationExample:
    return EvaluationExample(
        id=sample_id,
        prompt="",
        images=(),
        references=(answer,),
        choices=("A", "B", "C", "D"),
        metadata={"category": category, "l2_category": l2},
    )


def test_mmstar_extractor_matches_official_priority_rules() -> None:
    assert extract_mmstar_choice("The correct answer is (B).") == "B"
    assert extract_mmstar_choice("I considered A. but C. is correct") == "C"
    assert extract_mmstar_choice("I choose D") == "D"
    assert extract_mmstar_choice("no option") == ""


def test_mmstar_macro_averages_l2_capabilities() -> None:
    examples = [
        _example("1", "A", "perception", "scene"),
        _example("2", "A", "perception", "scene"),
        _example("3", "B", "reasoning", "logic"),
    ]
    score = MMStarProtocol("fixture").score(
        {"1": "A", "2": "B", "3": "B"},
        examples,
    )
    assert score.details["item_accuracy"] == 2 / 3
    assert score.details["l2_scores"] == {"logic": 1.0, "scene": 0.5}
    assert score.value == 0.75
