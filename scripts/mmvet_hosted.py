"""Submit a complete MM-Vet JSON to the pinned official hosted GPT-4.1 judge.

Run in an isolated CPU environment with gradio-client==2.6.1. Input is the
official {v1_0: answer, ...} format. This script never supplies an API key,
changes the model, or replaces missing/invalid grades with zeros.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path

SERVICE = "https://whyu-mm-vet-evaluator.hf.space"
SPACE_COMMIT = "ad769f03ae36b7a278bc3c72b912be892b2c4b36"
ANNOTATION_SHA = "6eced3121b9865098ea72aca233ca5ff7fd7b81374494d878815b9f856442d8e"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_grades(grades: dict, expected: set[str]) -> dict:
    if not isinstance(grades, dict) or set(grades) != expected:
        raise ValueError("judge did not return exactly the submitted sample IDs")
    values, models = {}, set()
    for sample_id, record in grades.items():
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), list) or len(record[key]) != 1
            for key in ("score", "content", "model")
        ):
            raise ValueError(f"expected one complete judge run: {sample_id}")
        score, content, model = record["score"][0], record["content"][0], record["model"][0]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError(f"invalid judge score: {sample_id}")
        try:
            parsed = float(str(content).strip().split(" ")[0])
        except (ValueError, IndexError) as error:
            raise ValueError(f"unparseable raw judge content: {sample_id}") from error
        if parsed != score or not isinstance(model, str) or not model.startswith("gpt-4.1"):
            raise ValueError(f"judge model/content/score mismatch: {sample_id}")
        models.add(model)
        values[sample_id] = float(score)
    if len(models) != 1 or not values:
        raise ValueError("mixed or empty judge model identity")
    return {
        "count": len(values),
        "value": sum(values.values()) / len(values),
        "per_item": values,
        "judge_model": next(iter(models)),
        "grading_runs": 1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not 60 <= args.timeout_seconds <= 3600:
        raise ValueError("timeout must be between 60 and 3600 seconds")
    if version("gradio-client") != "2.6.1":
        raise ValueError("use the qualified gradio-client==2.6.1 environment")
    if digest(args.annotations) != ANNOTATION_SHA:
        raise ValueError("MM-Vet annotations differ from the pinned v1 release")
    expected = set(json.loads(args.annotations.read_text()))
    answers = json.loads(args.submission.read_text())
    if (
        len(expected) != 218
        or set(answers) != expected
        or any(not isinstance(x, str) for x in answers.values())
    ):
        raise ValueError("submission must contain exactly 218 ID-keyed answer strings")
    with urllib.request.urlopen(
        "https://huggingface.co/api/spaces/whyu/MM-Vet_Evaluator", timeout=30
    ) as response:
        space = json.load(response)
    if space["sha"] != SPACE_COMMIT or space.get("runtime", {}).get("stage") != "RUNNING":
        raise ValueError(
            "official hosted service revision or readiness changed; review before submitting"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        path = args.output_dir / name
        with path.open("x") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")

    save(
        "request.json",
        {
            "service": SERVICE,
            "space_commit": SPACE_COMMIT,
            "judge": "gpt-4.1",
            "grading_runs": 1,
            "submission_sha256": digest(args.submission),
            "annotations_sha256": ANNOTATION_SHA,
            "client_version": version("gradio-client"),
            "script_sha256": digest(Path(__file__)),
            "input_format": "official MM-Vet v1 ID-to-answer JSON",
            "started_epoch": time.time(),
        },
    )
    from gradio_client import Client, handle_file

    job = None
    try:
        client = Client(SERVICE, verbose=False, download_files=str(args.output_dir / "downloads"))
        job = client.submit(
            handle_file(str(args.submission)), "", "gpt-4.1", "", api_name="/run_grade"
        )
        result = Path(job.result(timeout=args.timeout_seconds))
        with zipfile.ZipFile(result) as bundle:
            candidates = [
                name
                for name in bundle.namelist()
                if "-grade-1runs_" in name and name.endswith(".json")
            ]
            if len(candidates) != 1 or bundle.getinfo(candidates[0]).file_size > 10 * 1024**2:
                raise ValueError("unexpected official grading archive")
            grades = json.loads(bundle.read(candidates[0]))
        checked = checked_grades(grades, expected)
        save(
            "accepted-grade.json",
            {
                "status": "passed",
                **checked,
                "grading_zip": str(result),
                "grading_zip_sha256": digest(result),
                "request_sha256": digest(args.output_dir / "request.json"),
                "scope": "Official hosted GPT-4.1 one-run judging; "
                "separate from GPT-4-0613 published scores. "
                "The service exposes per-item model/content/score, not API response IDs.",
            },
        )
        print(json.dumps({key: value for key, value in checked.items() if key != "per_item"}))
    except BaseException as error:
        # Keep the request and any downloaded raw response for diagnosis.
        cancellation_error = None
        if job is not None:
            try:
                job.cancel()
            except Exception as cancel_error:
                cancellation_error = str(cancel_error)
        save(
            "failure.json",
            {
                "error_type": type(error).__name__,
                "message": str(error),
                "cancellation_error": cancellation_error,
                "status": "not_accepted",
            },
        )
        raise


if __name__ == "__main__":
    main()
