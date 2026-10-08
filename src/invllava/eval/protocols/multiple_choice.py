from __future__ import annotations

from invllava.eval.protocols.normalization import extract_choice
from invllava.eval.types import EvaluationExample, Score


class MultipleChoiceProtocol:
    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        correct = 0
        invalid = 0
        per_item: dict[str, int] = {}
        for example in examples:
            if len(example.references) != 1 or not example.choices:
                raise ValueError(f"multiple-choice example malformed: {example.id}")
            extracted = extract_choice(predictions[example.id], example.choices)
            is_correct = int(extracted == example.references[0].strip().upper())
            invalid += extracted is None
            correct += is_correct
            per_item[example.id] = is_correct
        return Score(
            value=correct / len(examples) if examples else 0.0,
            count=len(examples),
            details={"correct": correct, "invalid": invalid, "per_item": per_item},
        )
