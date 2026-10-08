from __future__ import annotations

import re
import string
from collections import defaultdict

from invllava.eval.types import EvaluationExample, Score

_VERBOSE_ANSWER = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")
_PUNCTUATION = ".()[],:;!*#{}"


def infer_mmbench_choice(answer: str, choices: tuple[str, ...]) -> str | None:
    """Deterministic VLMEvalKit-style option extraction, with no LLM judge fallback."""

    labels = tuple(string.ascii_uppercase[: len(choices)])
    answer = str(answer)
    if "Failed to obtain answer via API" in answer:
        return None
    refusal_fragments = (
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    )
    if any(fragment in answer for fragment in refusal_fragments):
        return None

    normalized = answer
    for character in _PUNCTUATION:
        normalized = normalized.replace(character, " ")
    tokens = [token.strip() for token in normalized.split()]
    present = [label for label in labels if label in tokens]
    if len(present) == 1 and tokens.index(present[0]) > len(tokens) - 5:
        return present[0]

    verbose_match = _VERBOSE_ANSWER.search(answer)
    if verbose_match and verbose_match.group(1).upper() in labels:
        return verbose_match.group(1).upper()

    answer_lower = answer.lower()
    choice_text = tuple(str(choice).lower() for choice in choices)
    if len(answer_lower) > 2 * sum(len(choice) for choice in choice_text):
        return None
    text_matches = [
        label for index, label in enumerate(labels) if choice_text[index] in answer_lower
    ]
    return text_matches[0] if len(text_matches) == 1 else None


class MMBenchCircularProtocol:
    """Group-level circular accuracy for public MMBench development annotations."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        groups: dict[str, list[EvaluationExample]] = defaultdict(list)
        for example in examples:
            if example.group_id is None:
                raise ValueError(f"MMBench example lacks circular group: {example.id}")
            if len(example.references) != 1 or not example.choices:
                raise ValueError(f"MMBench example malformed: {example.id}")
            groups[example.group_id].append(example)

        per_item: dict[str, int] = {}
        invalid_rotations = 0
        category_hits: dict[str, list[int]] = defaultdict(list)
        for group_id, rotations in groups.items():
            choice_counts = {len(rotation.choices) for rotation in rotations}
            if (
                len(choice_counts) != 1
                or not rotations
                or len(rotations) > len(rotations[0].choices)
            ):
                raise ValueError(
                    f"MMBench circular group {group_id} has an invalid "
                    f"{len(rotations)}-row/{len(rotations[0].choices)}-choice structure"
                )
            if group_id not in {rotation.id for rotation in rotations}:
                raise ValueError(f"MMBench circular group {group_id} lacks its base rotation")
            extracted = [
                infer_mmbench_choice(predictions[rotation.id], rotation.choices)
                for rotation in rotations
            ]
            invalid_rotations += sum(value is None for value in extracted)
            hit = int(
                all(
                    value == rotations[index].references[0].strip().upper()
                    for index, value in enumerate(extracted)
                )
            )
            per_item[group_id] = hit
            category = rotations[0].metadata.get("category")
            if category not in (None, ""):
                category_hits[str(category)].append(hit)

        correct = sum(per_item.values())
        category_accuracy = {
            category: sum(hits) / len(hits) for category, hits in sorted(category_hits.items())
        }
        return Score(
            value=correct / len(groups) if groups else 0.0,
            count=len(groups),
            details={
                "correct": correct,
                "circular_groups": len(groups),
                "rotations": len(examples),
                "invalid_rotations": invalid_rotations,
                "per_item": per_item,
                "category_accuracy": category_accuracy,
            },
        )
