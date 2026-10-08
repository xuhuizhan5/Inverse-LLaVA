from __future__ import annotations

from collections import defaultdict

from invllava.eval.types import EvaluationExample, Score


def extract_mme_answer(value: str) -> str | None:
    """Mirror pinned lmms-eval extraction, including its y/n extension.

    The original MME calculator does not accept single-letter answers or
    remove punctuation. Use scripts/audit_mme_official.py to check retained
    predictions against that independent calculator before paper comparison.
    """

    normalized = value.lower().strip().replace(".", "")
    if normalized in {"yes", "no"}:
        return normalized
    if len(normalized) == 1:
        return {"y": "yes", "n": "no"}.get(normalized)
    prefix = normalized[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return None


class MMEProtocol:
    def __init__(self, protocol_id: str, *, domain: str) -> None:
        self.id = protocol_id
        self.domain = domain

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        category_items: dict[str, list[tuple[str, int]]] = defaultdict(list)
        per_item: dict[str, int] = {}
        for example in examples:
            category = str(example.metadata["category"])
            example_domain = str(example.metadata.get("domain", ""))
            if example_domain != self.domain:
                raise ValueError(
                    f"MME {self.domain} scorer received {example_domain or 'unlabelled'} "
                    f"example {example.id}"
                )
            image_group = example.group_id or example.id
            prediction = extract_mme_answer(predictions[example.id])
            reference = example.references[0].strip().lower()
            if reference not in {"yes", "no"}:
                raise ValueError(f"MME reference for {example.id} is not binary")
            correct = int(prediction == reference)
            category_items[category].append((image_group, correct))
            per_item[example.id] = correct
        details: dict[str, float] = {}
        total = 0.0
        for category, items in sorted(category_items.items()):
            accuracy = sum(correct for _, correct in items) / len(items)
            by_image: dict[str, list[int]] = defaultdict(list)
            for image, correct in items:
                by_image[image].append(correct)
            invalid_groups = {
                image: len(values) for image, values in by_image.items() if len(values) != 2
            }
            if invalid_groups:
                first, count = next(iter(invalid_groups.items()))
                raise ValueError(f"MME group {first} has {count} questions; expected exactly 2")
            accuracy_plus = sum(all(values) for values in by_image.values()) / len(by_image)
            score = 100.0 * (accuracy + accuracy_plus)
            details[category] = score
            total += score
        return Score(
            total,
            len(examples),
            {"category_scores": details, "per_item": per_item},
        )
