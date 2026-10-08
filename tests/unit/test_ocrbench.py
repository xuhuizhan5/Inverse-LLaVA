from pathlib import Path

from invllava.eval.protocols.ocrbench import OCRBenchProtocol
from invllava.eval.types import EvaluationExample


def _example(sample_id: str, answer: str, dataset: str) -> EvaluationExample:
    return EvaluationExample(
        id=sample_id,
        prompt="question",
        images=(Path("fixture.png"),),
        references=(answer,),
        metadata={"dataset": dataset, "question_type": "Regular Text Recognition"},
    )


def test_ocrbench_matches_case_and_hme_whitespace_rules() -> None:
    examples = [_example("regular", "Hello World", "IIIT5K"), _example("hme", "x + y", "HME100k")]
    score = OCRBenchProtocol().score(
        {"regular": "The text is HELLO WORLD.", "hme": "x+y"},
        examples,
    )
    assert score.value == 1.0
    assert score.details["per_item"] == {"regular": 1, "hme": 1}
    assert score.details["category_count"] == {"Regular Text Recognition": 2}
    assert score.details["category_accuracy"] == {"Regular Text Recognition": 1.0}


def test_ocrbench_folds_fullwidth_ascii_like_pinned_upstream() -> None:
    example = _example("fullwidth", "ABC 123", "IIIT5K")
    fullwidth = "\uff21\uff22\uff23\u3000\uff11\uff12\uff13"
    score = OCRBenchProtocol().score({"fullwidth": fullwidth}, [example])
    assert score.value == 1.0
