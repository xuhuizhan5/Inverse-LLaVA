from __future__ import annotations

from collections import defaultdict

from invllava.eval.types import EvaluationExample, Score


def normalize_gqa_prediction(value: str) -> str:
    """Match LLaVA's GQA conversion before official exact-equality scoring."""

    return str(value).rstrip(".").lower()


class GQAProtocol:
    def __init__(self, protocol_id: str) -> None:
        self.id = protocol_id

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, int] = {}
        by_type: dict[str, dict[str, list[int]]] = {
            "structural": defaultdict(list),
            "semantic": defaultdict(list),
            "detailed": defaultdict(list),
        }
        by_answer_type: dict[str, list[int]] = defaultdict(list)
        for example in examples:
            if len(example.references) != 1:
                raise ValueError(f"GQA example requires one answer: {example.id}")
            correct = int(
                normalize_gqa_prediction(predictions[example.id]) == example.references[0]
            )
            per_item[example.id] = correct
            for family in by_type:
                value = example.metadata.get(f"{family}_type")
                if value not in (None, ""):
                    by_type[family][str(value)].append(correct)
            structural = example.metadata.get("structural_type")
            if structural not in (None, ""):
                by_answer_type["open" if structural == "query" else "binary"].append(correct)
        accuracy_by_type = {
            family: {name: sum(values) / len(values) for name, values in sorted(categories.items())}
            for family, categories in by_type.items()
            if categories
        }
        return Score(
            sum(per_item.values()) / len(per_item) if per_item else 0.0,
            len(per_item),
            {
                "per_item": per_item,
                "answer_type_accuracy": {
                    name: sum(values) / len(values)
                    for name, values in sorted(by_answer_type.items())
                },
                "accuracy_by_type": accuracy_by_type,
                "structural_accuracy": accuracy_by_type.get("structural", {}),
                "semantic_accuracy": accuracy_by_type.get("semantic", {}),
            },
        )
