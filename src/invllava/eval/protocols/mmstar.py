from __future__ import annotations

from collections import defaultdict

from invllava.eval.types import EvaluationExample, Score

_ANSWER_PHRASES = (
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    "옵션",
    "정답은",
    "답은",
    "답:",
    "答案是",
    "答案为",
    "选",
    "答えは",
)

_FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}


def extract_mmstar_choice(response: str, choices: tuple[str, ...] = ("A", "B", "C", "D")) -> str:
    """Mirror the MCQ extractor used by the frozen official MMStar task."""

    if not response or not response.strip():
        return ""
    text = response.strip().strip(",.!?;:'\"")
    text = f" {text} "
    candidates: list[tuple[str, int, str]] = []

    formats = (
        ("parentheses", lambda choice: f"({choice})"),
        ("period", lambda choice: f"{choice}."),
        ("colon", lambda choice: f"{choice}:"),
        ("right_paren", lambda choice: f"{choice})"),
        ("space", lambda choice: f"{choice} "),
    )
    for format_name, render in formats:
        for choice in choices:
            marker = render(choice)
            if marker in text:
                candidates.append((choice, text.rfind(marker), format_name))

    lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        phrase_index = lower.find(phrase)
        if phrase_index == -1:
            continue
        after = phrase_index + len(phrase)
        for choice in choices:
            choice_index = text.find(choice, after)
            if choice_index != -1:
                candidates.append((choice, choice_index, "phrase"))

    stripped = text.strip()
    for choice in choices:
        if stripped.startswith(choice) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((choice, 0, "start"))
        if stripped.endswith(choice) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((choice, len(text) - 1, "end"))

    if not candidates:
        for choice in choices:
            if choice in text:
                candidates.append((choice, text.rfind(choice), "fallback"))
    if not candidates:
        return ""
    candidates.sort(
        key=lambda item: (_FORMAT_PRIORITY.get(item[2], 0), item[1]),
        reverse=True,
    )
    return candidates[0][0]


class MMStarProtocol:
    """Official MMStar item scoring and macro-average over L2 capabilities."""

    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, int] = {}
        by_l2: dict[str, list[int]] = defaultdict(list)
        category_l2: dict[str, list[str]] = defaultdict(list)
        invalid = 0

        for example in examples:
            if len(example.references) != 1:
                raise ValueError(f"MMStar example has invalid target: {example.id}")
            category = str(example.metadata.get("category") or "").strip()
            l2_category = str(example.metadata.get("l2_category") or "").strip()
            if not category or not l2_category:
                raise ValueError(f"MMStar example lacks category metadata: {example.id}")
            extracted = extract_mmstar_choice(predictions[example.id])
            correct = int(extracted == example.references[0].strip().upper())
            invalid += extracted == ""
            per_item[example.id] = correct
            by_l2[l2_category].append(correct)
            if l2_category not in category_l2[category]:
                category_l2[category].append(l2_category)

        l2_scores = {category: sum(scores) / len(scores) for category, scores in by_l2.items()}
        category_scores = {
            category: sum(l2_scores[l2] for l2 in l2_names) / len(l2_names)
            for category, l2_names in category_l2.items()
        }
        value = sum(l2_scores.values()) / len(l2_scores) if l2_scores else 0.0
        correct = sum(per_item.values())
        return Score(
            value=value,
            count=len(examples),
            details={
                "correct": correct,
                "item_accuracy": correct / len(examples) if examples else 0.0,
                "invalid": invalid,
                "category_scores": category_scores,
                "l2_scores": l2_scores,
                "l2_counts": {category: len(scores) for category, scores in sorted(by_l2.items())},
                "per_item": per_item,
            },
        )
