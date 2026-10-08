from __future__ import annotations

from collections import defaultdict

from invllava.eval.types import EvaluationExample, Score


def _fold_fullwidth_ascii(value: str) -> str:
    folded: list[str] = []
    for character in value:
        codepoint = ord(character)
        if 0xFF01 <= codepoint <= 0xFF5E:
            folded.append(chr(codepoint - 0xFEE0))
        elif codepoint == 0x3000:
            folded.append(" ")
        else:
            folded.append(character)
    return "".join(folded)


class OCRBenchProtocol:
    """Rule-for-rule local OCRBench contains scoring.

    This mirrors the official LMMS task rule while avoiding its process-global
    category accumulator. Publication use remains gated on a golden comparison
    against the pinned upstream evaluator.
    """

    id = "ocrbench-official-contains-v1"

    @staticmethod
    def _correct(prediction: str, references: tuple[str, ...], dataset: str) -> int:
        if dataset == "HME100k":
            candidate = prediction.strip().replace("\n", " ").replace(" ", "")
            answers = [answer.strip().replace("\n", " ").replace(" ", "") for answer in references]
        else:
            candidate = _fold_fullwidth_ascii(prediction).lower().strip().replace("\n", " ")
            answers = [
                _fold_fullwidth_ascii(answer).lower().strip().replace("\n", " ")
                for answer in references
            ]
        return int(any(answer in candidate for answer in answers if answer))

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        per_item: dict[str, int] = {}
        category_correct: dict[str, int] = defaultdict(int)
        category_count: dict[str, int] = defaultdict(int)
        for example in examples:
            if not example.references:
                raise ValueError(f"OCRBench example has no answer: {example.id}")
            dataset = str(example.metadata.get("dataset", ""))
            category = str(example.metadata.get("question_type", ""))
            if not dataset or not category:
                raise ValueError(f"OCRBench metadata is incomplete: {example.id}")
            correct = self._correct(predictions[example.id], example.references, dataset)
            per_item[example.id] = correct
            category_correct[category] += correct
            category_count[category] += 1
        correct_total = sum(per_item.values())
        sorted_correct = dict(sorted(category_correct.items()))
        sorted_count = dict(sorted(category_count.items()))
        return Score(
            correct_total / len(examples) if examples else 0.0,
            len(examples),
            {
                "correct": correct_total,
                "category_correct": sorted_correct,
                "category_count": sorted_count,
                "category_accuracy": {
                    category: sorted_correct[category] / count
                    for category, count in sorted_count.items()
                },
                "per_item": per_item,
                "full_1000_item_score": correct_total if len(examples) == 1000 else None,
            },
        )
