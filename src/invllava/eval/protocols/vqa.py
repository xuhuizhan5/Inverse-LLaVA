from __future__ import annotations

from collections import defaultdict

from invllava.eval.protocols.evalai import (
    clean_vqa_text,
    normalize_evalai_answer,
    normalize_textvqa_answer,
)
from invllava.eval.protocols.normalization import normalize_answer
from invllava.eval.types import EvaluationExample, Score


class ExactMatchProtocol:
    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item = {
            example.id: int(
                normalize_answer(predictions[example.id])
                in {normalize_answer(reference) for reference in example.references}
            )
            for example in examples
        }
        return Score(
            sum(per_item.values()) / len(per_item) if per_item else 0.0,
            len(per_item),
            {"per_item": per_item},
        )


class VQAv2Protocol:
    """Official leave-one-annotator-out VQA consensus accuracy."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, float] = {}
        by_answer_type: dict[str, list[float]] = defaultdict(list)
        by_question_type: dict[str, list[float]] = defaultdict(list)
        for example in examples:
            answers = [clean_vqa_text(answer) for answer in example.references]
            _require_ten_answers(example, answers)
            prediction = clean_vqa_text(predictions[example.id])
            # This conditional mirrors the official VQAv2 evaluator exactly.
            if len(set(answers)) > 1:
                prediction = normalize_evalai_answer(prediction)
                answers = [normalize_evalai_answer(answer) for answer in answers]
            accuracy = _consensus_accuracy(prediction, answers)
            per_item[example.id] = accuracy
            by_answer_type[str(example.metadata.get("answer_type", "unknown"))].append(accuracy)
            by_question_type[str(example.metadata.get("question_type", "unknown"))].append(accuracy)
        return Score(
            sum(per_item.values()) / len(per_item) if per_item else 0.0,
            len(per_item),
            {
                "per_item": per_item,
                "answer_type_accuracy": {
                    key: sum(values) / len(values) for key, values in sorted(by_answer_type.items())
                },
                "question_type_accuracy": {
                    key: sum(values) / len(values)
                    for key, values in sorted(by_question_type.items())
                },
            },
        )


class TextVQAProtocol:
    """TextVQA's EvalAI-normalized ten-annotator consensus accuracy."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, float] = {}
        for example in examples:
            answers = [normalize_textvqa_answer(answer) for answer in example.references]
            _require_ten_answers(example, answers)
            prediction = normalize_textvqa_answer(predictions[example.id])
            per_item[example.id] = _consensus_accuracy(prediction, answers)
        return Score(
            sum(per_item.values()) / len(per_item) if per_item else 0.0,
            len(per_item),
            {"per_item": per_item},
        )


def _require_ten_answers(example: EvaluationExample, answers: list[str]) -> None:
    if len(answers) != 10:
        raise ValueError(
            f"VQA consensus example {example.id} has {len(answers)} references; expected 10"
        )


def _consensus_accuracy(prediction: str, answers: list[str]) -> float:
    leave_one_out = []
    for index in range(len(answers)):
        matches = sum(prediction == answer for j, answer in enumerate(answers) if j != index)
        leave_one_out.append(min(1.0, matches / 3.0))
    return sum(leave_one_out) / len(leave_one_out)
