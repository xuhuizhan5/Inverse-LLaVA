from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from invllava.artifacts.hashing import sha256_file
from invllava.eval.records import PredictionRecord
from invllava.eval.types import EvaluationExample


def _stable_rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def select_comparison_cases(
    inverse: list[PredictionRecord],
    baseline: list[PredictionRecord],
    inverse_correct: dict[str, bool],
    baseline_correct: dict[str, bool],
    *,
    per_group: int = 4,
    seed: int = 2026,
) -> dict[str, list[str]]:
    inverse_ids = {record.sample_id for record in inverse}
    baseline_ids = {record.sample_id for record in baseline}
    if inverse_ids != baseline_ids:
        raise ValueError("qualitative selection requires paired sample IDs")
    buckets: dict[str, list[str]] = defaultdict(list)
    for sample_id in inverse_ids:
        key = {
            (True, True): "both_correct",
            (True, False): "inverse_only",
            (False, True): "baseline_only",
            (False, False): "both_wrong",
        }[(inverse_correct[sample_id], baseline_correct[sample_id])]
        buckets[key].append(sample_id)
    return {
        key: sorted(values, key=lambda value: _stable_rank(value, seed))[:per_group]
        for key, values in sorted(buckets.items())
    }


def select_multimodel_cases(
    correct_by_model: dict[str, dict[str, bool]],
    *,
    primary_model: str,
    per_group: int = 4,
    seed: int = 2026,
) -> tuple[dict[str, list[str]], dict[str, dict[str, bool]], dict[str, int]]:
    """Select disjoint correctness patterns across one method and comparison baselines."""

    labels = list(correct_by_model)
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("qualitative selection requires at least two unique model labels")
    if primary_model not in correct_by_model:
        raise ValueError("primary model is absent from qualitative inputs")
    if per_group <= 0:
        raise ValueError("per-group must be positive")
    id_sets = {frozenset(values) for values in correct_by_model.values()}
    if len(id_sets) != 1 or not next(iter(id_sets), frozenset()):
        raise ValueError("qualitative model scores must have identical non-empty sample IDs")

    baselines = [label for label in labels if label != primary_model]
    buckets: dict[str, list[str]] = defaultdict(list)
    patterns: dict[str, dict[str, bool]] = {}
    for sample_id in next(iter(correct_by_model.values())):
        pattern = {label: correct_by_model[label][sample_id] for label in labels}
        patterns[sample_id] = pattern
        primary = pattern[primary_model]
        baseline_values = [pattern[label] for label in baselines]
        if primary and all(baseline_values):
            bucket = "all_correct"
        elif not primary and not any(baseline_values):
            bucket = "all_wrong"
        elif primary and not any(baseline_values):
            bucket = f"{primary_model}_only"
        elif not primary and all(baseline_values):
            bucket = "baselines_only"
        elif primary:
            bucket = f"{primary_model}_with_baseline_disagreement"
        else:
            bucket = "baseline_disagreement"
        buckets[bucket].append(sample_id)
    selected = {
        key: sorted(values, key=lambda value: _stable_rank(value, seed))[:per_group]
        for key, values in sorted(buckets.items())
    }
    selected_patterns = {
        sample_id: patterns[sample_id] for values in selected.values() for sample_id in values
    }
    group_counts = {key: len(values) for key, values in sorted(buckets.items())}
    return selected, selected_patterns, group_counts


ERROR_LABELS = (
    "perception",
    "ocr",
    "localization",
    "count",
    "language_prior_shortcut",
    "hallucination",
    "format",
)


def materialize_case_images(
    examples: dict[str, EvaluationExample],
    sample_ids: list[str],
    destination: str | Path,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Copy selected case images into an immutable, content-addressed directory."""

    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    paths_by_sample: dict[str, list[str]] = {}
    inventory_by_file: dict[str, dict[str, Any]] = {}
    try:
        for sample_id in sample_ids:
            if sample_id not in examples:
                raise ValueError(f"selected case is absent from examples: {sample_id}")
            source_paths = examples[sample_id].images
            if not source_paths:
                raise ValueError(f"selected visual case has no image: {sample_id}")
            copied: list[str] = []
            for source in source_paths:
                if not source.is_file():
                    raise FileNotFoundError(source)
                digest = sha256_file(source)
                suffix = source.suffix.lower() or ".img"
                relative = f"{digest}{suffix}"
                target = temporary / relative
                if not target.exists():
                    shutil.copyfile(source, target)
                    if sha256_file(target) != digest:
                        raise RuntimeError(f"case image copy failed verification: {source}")
                copied.append(str(output / relative))
                record = inventory_by_file.setdefault(
                    relative,
                    {
                        "file": relative,
                        "sha256": digest,
                        "size_bytes": target.stat().st_size,
                        "sample_ids": [],
                    },
                )
                record["sample_ids"].append(sample_id)
            paths_by_sample[sample_id] = copied
        temporary.chmod(0o755)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    inventory = [inventory_by_file[key] for key in sorted(inventory_by_file)]
    return paths_by_sample, inventory
