from __future__ import annotations

import hashlib
import math
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.eval.protocols.evalai import clean_vqa_text, normalize_textvqa_answer
from invllava.eval.records import PredictionRecord
from invllava.eval.validation import validate_coverage


def parse_vqav2_result(payload: object, *, split: str = "test-dev") -> dict[str, float]:
    """Validate EvalAI's split-keyed aggregate, retaining percentage units.

    This parses a returned score, not submission authenticity or correctness.
    The caller must bind the raw response to its uploaded predictions and phase.
    Aggregate-only results cannot supply per-question confidence intervals.
    """
    if split not in {"test-dev", "test-standard"}:
        raise ValueError("unsupported VQAv2 result split")
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], dict)
        or set(payload[0]) != {split}
    ):
        raise ValueError("expected one official result for the specified VQAv2 split")
    metrics = payload[0][split]
    names = {"yes/no", "number", "other", "overall"}
    if not isinstance(metrics, dict) or set(metrics) != names:
        raise ValueError("VQAv2 result metric names changed or are incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100
        for value in metrics.values()
    ):
        raise ValueError("VQAv2 scores must be finite percentages in [0, 100]")
    return {name: float(metrics[name]) for name in sorted(names)}


def submission_checkpoint_digest(identity: str, snapshot: str | Path | None = None) -> str:
    """Resolve a saved delta digest or verify a local Hub snapshot's full inventory.

    HF reference manifests retain ``local_path@revision``. Packaging binds those
    exact bytes without rewriting the inference manifest or its checkpoint ID.
    """
    if len(identity) == 64 and all(character in "0123456789abcdef" for character in identity):
        if snapshot is not None:
            raise ValueError("a delta checkpoint already has a content digest")
        return identity
    if snapshot is None or "@" not in identity:
        raise ValueError("HF reference packaging requires --checkpoint-snapshot")
    from invllava.artifacts.hub_snapshot import verify_hub_snapshot
    from invllava.config.identifiers import canonical_json

    locator, revision = identity.rsplit("@", 1)
    if not Path(locator).is_absolute() or Path(locator).resolve() != Path(snapshot).resolve():
        raise ValueError("snapshot path differs from the evaluated model locator")
    inventory = verify_hub_snapshot(snapshot)
    if inventory["resolved_revision"] != revision:
        raise ValueError("snapshot revision differs from the evaluated model revision")
    payload = {key: inventory[key] for key in ("repo_id", "resolved_revision", "files")}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_vqav2_submission(
    records: list[PredictionRecord],
    expected_question_ids: set[str],
    destination: str | Path,
    *,
    full_test_question_ids: list[int] | None = None,
) -> None:
    """Package exact predictions; optionally reproduce LLaVA's full-test envelope.

    The LLaVA test-dev exporter normalizes generated answers with its M4C
    processor and inserts empty answers only for IDs outside the scored subset.
    Such an envelope does not establish a full-test or test-standard result.
    """
    validate_coverage(records, expected_question_ids)
    if full_test_question_ids is not None:
        if (
            len(full_test_question_ids) != len(set(full_test_question_ids))
            or any(type(item) is not int for item in full_test_question_ids)
            or not {int(item) for item in expected_question_ids}.issubset(full_test_question_ids)
        ):
            raise ValueError("full-test envelope must uniquely cover every evaluated question ID")
    rows = []
    for record in records:
        if record.references:
            raise ValueError("test-dev submission records must not contain hidden references")
        try:
            question_id = int(record.sample_id)
        except ValueError as error:
            raise ValueError(f"VQAv2 question id is not an integer: {record.sample_id}") from error
        normalize = (
            normalize_textvqa_answer if full_test_question_ids is not None else clean_vqa_text
        )
        rows.append({"question_id": question_id, "answer": normalize(record.prediction)})
    if full_test_question_ids is not None:
        answers = {row["question_id"]: row["answer"] for row in rows}
        rows = [
            {"question_id": key, "answer": answers.get(key, "")} for key in full_test_question_ids
        ]
    else:
        rows.sort(key=lambda row: row["question_id"])
    atomic_write_json(destination, rows)


def build_mmvet_submission(
    records: list[PredictionRecord], expected_question_ids: set[str], destination: str | Path
) -> None:
    """Preserve raw answers and official IDs for external MM-Vet judging."""
    validate_coverage(records, expected_question_ids)
    if expected_question_ids != {f"v1_{index}" for index in range(218)}:
        raise ValueError("MM-Vet v1 submission requires all 218 official IDs")
    if any(not isinstance(record.prediction, str) for record in records):
        raise ValueError("MM-Vet answers must be strings")
    answers = {record.sample_id: record.prediction for record in records}
    atomic_write_json(destination, {f"v1_{index}": answers[f"v1_{index}"] for index in range(218)})


def write_submission_manifest(destination: str | Path, manifest: dict[str, object]) -> None:
    required = {"checkpoint_sha256", "protocol_id", "dataset_split"}
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"submission manifest missing: {sorted(missing)}")
    source_digest = manifest.get("execution_source_sha256")
    if not manifest.get("code_commit") and not source_digest:
        raise ValueError(
            "submission manifest requires a clean code commit or execution-source SHA-256"
        )
    if source_digest is not None and (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise ValueError("execution-source identity must be a lowercase SHA-256 digest")
    atomic_write_json(destination, manifest)
