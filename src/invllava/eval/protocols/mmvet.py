"""Import the pinned official hosted MM-Vet grades without calling a judge."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

from invllava.artifacts.hashing import sha256_file
from invllava.eval.types import EvaluationExample, Score


class HostedMMVetProtocol:
    def __init__(self, protocol_id: str, grading_dir: str | Path, submission: str | Path):
        self.id = protocol_id
        self.grading_dir = Path(grading_dir)
        self.submission = Path(submission)

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score:
        expected = {f"v1_{index}" for index in range(218)}
        if (
            len(examples) != 218
            or set(predictions) != expected
            or {row.id for row in examples} != expected
        ):
            raise ValueError("MM-Vet judging requires exactly the 218 v1 questions")
        if json.loads(self.submission.read_text()) != predictions:
            raise ValueError("submitted answers differ from these model predictions")
        request_path = self.grading_dir / "request.json"
        request = json.loads(request_path.read_text())
        accepted_path = self.grading_dir / "accepted-grade.json"
        accepted = json.loads(accepted_path.read_text())
        if request["submission_sha256"] != sha256_file(self.submission) or accepted[
            "request_sha256"
        ] != sha256_file(request_path):
            raise ValueError("judge receipt does not bind the submitted answer bytes")
        if (
            request["space_commit"] != "ad769f03ae36b7a278bc3c72b912be892b2c4b36"
            or request["annotations_sha256"]
            != "6eced3121b9865098ea72aca233ca5ff7fd7b81374494d878815b9f856442d8e"
            or request["judge"] != "gpt-4.1"
            or accepted["judge_model"] != "gpt-4.1-2025-04-14"
            or accepted["status"] != "passed"
            or accepted["grading_runs"] != 1
            or request["grading_runs"] != 1
            or accepted["count"] != 218
        ):
            raise ValueError("judge/reference identity differs from the qualified hosted protocol")
        archives = [
            path
            for path in self.grading_dir.rglob("*.zip")
            if sha256_file(path) == accepted["grading_zip_sha256"]
        ]
        if len(archives) != 1:
            raise ValueError("one unchanged raw grading ZIP is required")
        with zipfile.ZipFile(archives[0]) as bundle:
            names = [
                name
                for name in bundle.namelist()
                if "-grade-1runs_" in name and name.endswith(".json")
            ]
            if len(names) != 1 or bundle.getinfo(names[0]).file_size > 10 * 1024**2:
                raise ValueError("unexpected raw grading archive")
            grades = json.loads(bundle.read(names[0]))
        if (
            not isinstance(grades, dict)
            or set(grades) != expected
            or set(accepted["per_item"]) != expected
        ):
            raise ValueError("raw or accepted grades have incomplete question coverage")
        per_item = {}
        for sample_id, record in grades.items():
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), list) or len(record[key]) != 1
                for key in ("score", "content", "model")
            ):
                raise ValueError(f"incomplete judge record: {sample_id}")
            value = record["score"][0]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"invalid judge value: {sample_id}")
            try:
                parsed = float(str(record["content"][0]).strip().split(" ")[0])
            except ValueError as error:
                raise ValueError(f"invalid judge content: {sample_id}") from error
            if (
                parsed != value
                or record["model"][0] != accepted["judge_model"]
                or value != accepted["per_item"][sample_id]
            ):
                raise ValueError(f"raw and accepted judge records disagree: {sample_id}")
            per_item[sample_id] = float(value)
        value = sum(per_item.values()) / 218
        if (
            isinstance(accepted["value"], bool)
            or not isinstance(accepted["value"], (int, float))
            or not math.isfinite(accepted["value"])
            or abs(value - accepted["value"]) > 1e-12
        ):
            raise ValueError("accepted aggregate differs from the raw item grades")
        return Score(
            value=value,
            count=218,
            details={
                "per_item": per_item,
                "judge_model": accepted["judge_model"],
                "grading_runs": 1,
                "grading_zip_sha256": accepted["grading_zip_sha256"],
                "accepted_grade_sha256": sha256_file(accepted_path),
                "request_sha256": sha256_file(request_path),
                "submission_sha256": sha256_file(self.submission),
                "scope": "Official hosted judging; one run does not estimate judge-run variance.",
            },
        )
