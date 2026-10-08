from __future__ import annotations

from collections import Counter

from invllava.eval.records import PredictionRecord


def validate_coverage(records: list[PredictionRecord], expected_ids: set[str]) -> None:
    actual = [record.sample_id for record in records]
    duplicates = {sample_id for sample_id, count in Counter(actual).items() if count > 1}
    missing = expected_ids - set(actual)
    extra = set(actual) - expected_ids
    if duplicates or missing or extra:
        raise ValueError(
            f"prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}, "
            f"duplicates={len(duplicates)}"
        )


def validate_no_reference_leakage(records: list[PredictionRecord]) -> None:
    for record in records:
        forbidden_keys = {"answer", "answers", "reference", "references", "ground_truth", "label"}
        leaked = forbidden_keys.intersection(key.lower() for key in record.metadata)
        if leaked:
            raise ValueError(
                f"reference-bearing metadata in sample {record.sample_id}: {sorted(leaked)}"
            )


def validate_frozen_prompts(records: list[PredictionRecord], examples: list[object]) -> None:
    expected = {str(example.id): str(example.prompt) for example in examples}
    for record in records:
        if record.sample_id not in expected or record.prompt != expected[record.sample_id]:
            raise ValueError(f"prompt mismatch for sample {record.sample_id}")
