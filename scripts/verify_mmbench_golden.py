#!/usr/bin/env python3
"""Verify MMBench data, prompts, bindings, extraction, and circular scoring."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import sys
import types
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file

_DEFAULT_ANSWER_MEMBER = "mmbench/answers/mmbench_dev_20230712/llava-v1.5-13b.jsonl"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _load_upstream(path: Path) -> Any:
    logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    vlmeval = types.ModuleType("vlmeval")
    smp = types.ModuleType("vlmeval.smp")
    log = types.ModuleType("vlmeval.smp.log")
    log.get_logger = lambda *_args, **_kwargs: logger
    sys.modules.setdefault("vlmeval", vlmeval)
    sys.modules.setdefault("vlmeval.smp", smp)
    sys.modules.setdefault("vlmeval.smp.log", log)
    spec = importlib.util.spec_from_file_location("pinned_matching_util", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned MMBench matching utility: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _choices(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[label]) for label in "ABCD" if row.get(label) not in (None, "", "nan"))


def _resolved_compact_images(rows: list[dict[str, str]]) -> dict[str, str]:
    result = {}
    for row in rows:
        group_id = str(int(row["index"]) % 1_000_000)
        value = str(row["image"])
        if not value.isdigit():
            result[group_id] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--benchmark-id", choices=("mmbench-en", "mmbench-cn"), required=True)
    parser.add_argument("--upstream-matching-util", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--compact-tsv", type=Path, required=True)
    parser.add_argument("--legacy-tsv", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--llava-archive", type=Path)
    parser.add_argument("--llava-answer-member", default=_DEFAULT_ANSWER_MEMBER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    sys.path.insert(0, str(args.source))
    from invllava.eval.datasets import load_examples
    from invllava.eval.protocols.mmbench import MMBenchCircularProtocol, infer_mmbench_choice
    from invllava.prompting import format_vicuna_v1_user_prompt

    upstream = _load_upstream(args.upstream_matching_util)
    compact = _load_rows(args.compact_tsv)
    legacy = _load_rows(args.legacy_tsv)
    if len(compact) != len(legacy) or len(compact) != 4_329:
        raise ValueError("official MMBench v1.0 files must contain 4,329 rotations")
    non_image_fields = tuple(field for field in compact[0] if field != "image")
    field_mismatches = []
    image_payload_mismatches = []
    compact_images = _resolved_compact_images(compact)
    for compact_row, legacy_row in zip(compact, legacy, strict=True):
        sample_id = str(compact_row["index"])
        if any(compact_row[field] != legacy_row[field] for field in non_image_fields):
            field_mismatches.append(sample_id)
        group_id = str(int(sample_id) % 1_000_000)
        if compact_images.get(group_id) != legacy_row["image"]:
            image_payload_mismatches.append(sample_id)

    examples = load_examples(args.examples)
    example_index = {example.id: example for example in examples}
    row_index = {str(row["index"]): row for row in compact}
    prompt_mismatches = []
    reference_mismatches = []
    group_mismatches = []
    image_mismatches = []
    group_counts = Counter(str(example.group_id) for example in examples)
    for sample_id, row in row_index.items():
        example = example_index[sample_id]
        choices = _choices(row)
        hint = row.get("hint")
        hint_block = f"Hint: {hint}\n" if hint not in (None, "", "nan") else ""
        expected_prompt = format_vicuna_v1_user_prompt(
            "<image>\n"
            f"{hint_block}Question: {row['question']}\n"
            "Options:\n"
            + "\n".join(
                f"{label}. {choice}"
                for label, choice in zip("ABCD"[: len(choices)], choices, strict=True)
            )
            + "\nPlease select the correct answer from the options above."
        )
        if example.prompt != expected_prompt:
            prompt_mismatches.append(sample_id)
        if example.references != (str(row["answer"]).upper(),) or example.choices != choices:
            reference_mismatches.append(sample_id)
        group_id = str(int(sample_id) % 1_000_000)
        if example.group_id != group_id or group_counts[group_id] > len(choices):
            group_mismatches.append(sample_id)
        expected_payload = base64.b64decode(compact_images[group_id], validate=True)
        if (
            len(example.images) != 1
            or not example.images[0].is_file()
            or hashlib.sha256(example.images[0].read_bytes()).digest()
            != hashlib.sha256(expected_payload).digest()
        ):
            image_mismatches.append(sample_id)

    released = []
    if args.llava_archive is not None:
        with zipfile.ZipFile(args.llava_archive) as bundle:
            released = [
                json.loads(line)
                for line in bundle.read(args.llava_answer_member).decode("utf-8").splitlines()
            ]
    predictions = {
        str(row["question_id"]): str(row["text"])
        for row in released
        if str(row["question_id"]) in example_index
    }
    extraction_mismatches = []
    candidates = (
        "A",
        "a",
        "b",
        "c",
        "d",
        "The answer is B.",
        "red",
        "A or B",
        "Cannot determine the answer",
        "",
    )
    comparisons = 0
    for example in examples:
        choice_map = {
            label: choice
            for label, choice in zip("ABCD"[: len(example.choices)], example.choices, strict=True)
        }
        values = candidates + ((predictions[example.id],) if example.id in predictions else ())
        for value in values:
            official = upstream.can_infer(value, dict(choice_map))
            official = official if official in choice_map else None
            local = infer_mmbench_choice(value, example.choices)
            comparisons += 1
            if official != local:
                extraction_mismatches.append(
                    {
                        "sample_id": example.id,
                        "prediction": value,
                        "upstream": official,
                        "local": local,
                    }
                )

    released_score = None
    if len(predictions) == len(examples):
        released_score = MMBenchCircularProtocol("golden").score(predictions, examples).value
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    count_contract = (
        len(examples) == len(example_index) == 4_329
        and len(group_counts) == 1_164
        and manifest.get("sample_count") == 4_329
        and manifest.get("group_count") == 1_164
        and manifest.get("image_integrity", {}).get("decoded_images") == 1_164
        and manifest.get("image_integrity", {}).get("passed") is True
    )
    mismatch_count = sum(
        len(values)
        for values in (
            field_mismatches,
            image_payload_mismatches,
            prompt_mismatches,
            reference_mismatches,
            group_mismatches,
            image_mismatches,
            extraction_mismatches,
        )
    )
    payload = {
        "format": "invllava-upstream-scorer-golden-v1",
        "benchmark": args.benchmark_id,
        "verifier_sha256": sha256_file(Path(__file__)),
        "upstream_revision": args.upstream_revision,
        "upstream_matching_util_sha256": sha256_file(args.upstream_matching_util),
        "compact_tsv_sha256": sha256_file(args.compact_tsv),
        "legacy_tsv_sha256": sha256_file(args.legacy_tsv),
        "llava_archive_sha256": (
            sha256_file(args.llava_archive) if args.llava_archive is not None else None
        ),
        "llava_answer_member": args.llava_answer_member if args.llava_archive else None,
        "examples_sha256": sha256_file(args.examples),
        "manifest_sha256": sha256_file(args.manifest),
        "sample_count": len(examples),
        "group_count": len(group_counts),
        "extraction_comparisons": comparisons,
        "released_llava_13b_coverage": len(predictions),
        "released_llava_13b_circular_exact_accuracy": released_score,
        "mismatch_counts": {
            "non_image_fields": len(field_mismatches),
            "compact_image_payloads": len(image_payload_mismatches),
            "prompts": len(prompt_mismatches),
            "references": len(reference_mismatches),
            "groups": len(group_mismatches),
            "image_bindings": len(image_mismatches),
            "extraction": len(extraction_mismatches),
        },
        "mismatch_examples": {
            "fields": field_mismatches[:20],
            "images": image_payload_mismatches[:20],
            "prompts": prompt_mismatches[:20],
            "references": reference_mismatches[:20],
            "groups": group_mismatches[:20],
            "bindings": image_mismatches[:20],
            "extraction": extraction_mismatches[:20],
        },
        "count_contract_passed": count_contract,
        "passed": mismatch_count == 0 and count_contract,
    }
    atomic_write_json(args.output, payload)
    if not payload["passed"]:
        raise RuntimeError(f"MMBench golden failed with {mismatch_count} mismatches")
    print(
        "MMBench golden passed: "
        f"{len(examples)} rotations, {len(group_counts)} groups, {comparisons} extractor cases"
    )


if __name__ == "__main__":
    main()
