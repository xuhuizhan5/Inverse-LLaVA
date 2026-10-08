from invllava.eval.protocols.evalai import normalize_evalai_answer, normalize_textvqa_answer
from invllava.eval.protocols.vqa import TextVQAProtocol, VQAv2Protocol
from invllava.eval.types import EvaluationExample


def test_evalai_normalization_handles_numbers_articles_and_contractions() -> None:
    assert normalize_evalai_answer("The TWO, cats!") == "2 cats"
    assert normalize_evalai_answer("Dont") == "don't"
    assert normalize_evalai_answer("1,000.5") == "1000.5"


def test_evalai_normalization_preserves_upstream_apostrophe_quirks() -> None:
    assert normalize_evalai_answer("cat's") == "cat's"
    assert normalize_evalai_answer("Id've") == "id've"


def test_textvqa_uses_evalai_consensus() -> None:
    example = EvaluationExample("1", "", (), ("two",) * 4 + ("three",) * 6)
    score = TextVQAProtocol("fixture").score({"1": "2"}, [example])
    assert score.value == 1.0


def test_textvqa_preserves_paper_evaluator_edge_cases() -> None:
    assert normalize_textvqa_answer("red,blue?") == "redblue"
    assert normalize_textvqa_answer("Im") == "im"
    assert normalize_textvqa_answer("The TWO, cats!") == "2 cats"


def test_vqav2_preserves_official_unanimous_answer_branch() -> None:
    example = EvaluationExample("1", "", (), ("Two",) * 10)
    # The official VQAv2 evaluator skips normalization when all raw GT answers agree.
    assert VQAv2Protocol("fixture").score({"1": "Two"}, [example]).value == 1.0
    assert VQAv2Protocol("fixture").score({"1": "two"}, [example]).value == 0.0
