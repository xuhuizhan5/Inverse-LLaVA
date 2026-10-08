from __future__ import annotations

import re
import string

from invllava.eval.types import EvaluationExample, Score

_LEADING_OPTION = re.compile(r"^\s*([A-Z])\.")


def filter_ai2d_response(response: str) -> str:
    """Apply the pinned lmms-eval AI2D response filter."""

    match = _LEADING_OPTION.match(response)
    return match.group(1) if match else response


def normalize_ai2d_exact(value: str) -> str:
    """Apply lm-eval exact-match case and ASCII-punctuation normalization."""

    normalized = "".join(character for character in value if character not in string.punctuation)
    return normalized.lower().strip()


class AI2DProtocol:
    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, int] = {}
        invalid = 0
        for example in examples:
            if len(example.references) != 1 or not example.choices:
                raise ValueError(f"AI2D example malformed: {example.id}")
            filtered = filter_ai2d_response(predictions[example.id])
            normalized = normalize_ai2d_exact(filtered)
            target = normalize_ai2d_exact(example.references[0])
            per_item[example.id] = int(normalized == target)
            valid = {
                normalize_ai2d_exact(chr(ord("A") + index)) for index in range(len(example.choices))
            }
            invalid += normalized not in valid
        correct = sum(per_item.values())
        return Score(
            value=correct / len(examples) if examples else 0.0,
            count=len(examples),
            details={"correct": correct, "invalid": invalid, "per_item": per_item},
        )
