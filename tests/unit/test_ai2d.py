from pathlib import Path

from invllava.eval.protocols.ai2d import AI2DProtocol, filter_ai2d_response
from invllava.eval.types import EvaluationExample


def _example() -> EvaluationExample:
    return EvaluationExample(
        id="one",
        prompt="prompt",
        images=(Path("image.png"),),
        references=("B",),
        choices=("left", "right"),
    )


def test_ai2d_filter_matches_pinned_leading_option_rule() -> None:
    assert filter_ai2d_response(" B. right") == "B"
    assert filter_ai2d_response("The answer is B.") == "The answer is B."


def test_ai2d_protocol_uses_official_exact_match_normalization() -> None:
    protocol = AI2DProtocol("fixture")

    direct = protocol.score({"one": "b)"}, [_example()])
    sentence = protocol.score({"one": "The answer is B."}, [_example()])

    assert direct.value == 1.0
    assert sentence.value == 0.0
    assert sentence.details["invalid"] == 1
