from __future__ import annotations

import re

from invllava.eval.types import EvaluationExample, Score


def extract_llava_scienceqa_choice(value: str, choice_count: int) -> str | None:
    """Reproduce LLaVA-1.5's ScienceQA answer parsing exactly."""

    if type(choice_count) is not int or not 1 <= choice_count <= 5:
        raise ValueError("official ScienceQA requires one to five choices")
    # The official parser first recognizes A--E, then checks available choices.
    # Preserve raw whitespace and branch precedence, even for invalid prefixes.
    options = tuple("ABCDE")
    prediction = value
    if prediction in options:
        answer = prediction
    elif len(prediction) >= 3 and prediction[0] in options and prediction[1:3] == ". ":
        answer = prediction[0]
    else:
        matches = re.findall(r"The answer is ([A-Z]).", prediction)
        answer = matches[0] if len(matches) == 1 else None
    return answer if answer in options[:choice_count] else None


class ScienceQAProtocol:
    """Image-only accuracy from LLaVA's official ScienceQA evaluation script."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        correct = 0
        invalid = 0
        per_item: dict[str, int] = {}
        for example in examples:
            if len(example.references) != 1 or not example.choices:
                raise ValueError(f"ScienceQA example malformed: {example.id}")
            extracted = extract_llava_scienceqa_choice(
                predictions[example.id], len(example.choices)
            )
            is_correct = int(extracted == example.references[0])
            invalid += extracted is None
            correct += is_correct
            per_item[example.id] = is_correct
        return Score(
            value=correct / len(examples) if examples else 0.0,
            count=len(examples),
            details={"correct": correct, "invalid": invalid, "per_item": per_item},
        )
