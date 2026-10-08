#!/usr/bin/env python3
"""Measure retained supervision with the training tokenizer; never load images.

Reports both truncation boundaries, all-masked rows, and source-level target
coverage. This audits the frozen data policy without excluding or changing rows.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.data.controls import supervised_tokens_after_expansion


def supervision_summary(input_ids, labels, *, image_patch_counts, max_length):
    """Count causal targets before collation, after collation, and after expansion."""
    surviving = supervised_tokens_after_expansion(
        input_ids, labels, image_patch_counts=image_patch_counts, max_length=max_length
    )
    raw = sum(
        i > 0 and token != -200 and label != -100
        for i, (token, label) in enumerate(zip(input_ids, labels, strict=True))
    )
    collated = sum(
        i > 0 and token != -200 and label != -100
        for i, (token, label) in enumerate(
            zip(input_ids[:max_length], labels[:max_length], strict=True)
        )
    )
    expanded = min(len(input_ids), max_length) + sum(n - 1 for n in image_patch_counts)
    return {
        "encoded_tokens": len(input_ids),
        "expanded_tokens_before_final_truncation": expanded,
        "truncated_at_collation": int(len(input_ids) > max_length),
        "truncated_after_expansion": int(expanded > max_length),
        "raw_targets": raw,
        "collated_targets": collated,
        "surviving_targets": surviving,
        "raw_zero_targets": int(raw == 0),
        "surviving_zero_targets": int(surviving == 0),
        "targets_lost_at_collation": raw - collated,
        "targets_lost_after_expansion": collated - surviving,
        **{f"rows_below_{n}_targets": int(surviving < n) for n in (16, 32, 64)},
    }


def main():
    from transformers import CLIPVisionConfig

    from invllava.config.identifiers import content_id, scientific_payload
    from invllava.config.loader import ConfigRepository
    from invllava.data.collate import encode_vicuna_v1
    from invllava.data.manifest import PreparedDatasetManifest
    from invllava.data.types import ConversationSample, Turn
    from invllava.model.loaders import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum", type=int, help="prefix timing diagnostic only")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.maximum is not None and args.maximum <= 0:
        raise ValueError("--maximum must be positive")
    config = ConfigRepository().resolve(args.experiment)
    manifest = PreparedDatasetManifest.read(args.manifest)
    manifest.verify(
        args.data,
        expected_data_id=config.data.id,
        expected_source_revision=config.data.annotation.revision,
        expected_source_sha256=config.data.annotation.sha256,
        expected_source_filter=config.data.include_sources,
    )
    before_sha = sha256_file(args.data)
    corpus_rows = int(manifest.audit["samples"])
    tokenizer = load_tokenizer(config.model, local_files_only=True)
    vision = CLIPVisionConfig.from_pretrained(
        config.model.vision.checkpoint,
        revision=config.model.vision.revision,
        local_files_only=True,
    )
    patches = (config.model.vision.image_size // vision.patch_size) ** 2
    patches += int(config.model.vision.feature_select == "cls_patch")
    groups = defaultdict(Counter)
    longest = defaultdict(list)
    failures = []
    failed_count = 0
    limit = min(args.maximum or corpus_rows, corpus_rows)
    started = time.monotonic()

    # Image paths are inert metadata here. The prepared manifest already binds
    # their decode audit; resolving millions of paths would add unrelated I/O.
    def rows():
        with args.data.open() as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    sample = ConversationSample(
                        id=str(value["id"]),
                        source=str(value["source"]),
                        images=tuple(Path(path) for path in value["images"]),
                        turns=tuple(
                            Turn(str(turn["role"]), str(turn["text"])) for turn in value["turns"]
                        ),
                    )
                    sample.validate()
                    yield sample

    processed = 0
    for index, sample in enumerate(rows()):
        if index >= limit:
            if limit == corpus_rows:
                raise RuntimeError("annotation has more rows than its manifest")
            break
        processed += 1
        group = groups[sample.source]
        group["rows"] += 1
        try:
            ids, labels = encode_vicuna_v1(tokenizer, sample.turns)
            summary = supervision_summary(
                ids,
                labels,
                image_patch_counts=[patches] * len(sample.images),
                max_length=config.model.language.max_length,
            )
        except ValueError as error:
            group["row_contract_errors"] += 1
            failed_count += 1
            if len(failures) < 50:
                failures.append({"id": sample.id, "source": sample.source, "error": str(error)})
        else:
            group.update(summary)
            candidates = longest[sample.source]
            candidates.append({"id": sample.id, **summary})
            candidates.sort(key=lambda row: (-row["encoded_tokens"], row["id"]))
            del candidates[5:]
        if (index + 1) % 5000 == 0 or index + 1 == limit:
            print(f"{index + 1}/{limit} rows; {time.monotonic() - started:.1f}s", flush=True)
    if processed != limit:
        raise RuntimeError("annotation has fewer rows than its manifest")
    if sha256_file(args.data) != before_sha:
        raise RuntimeError("annotation changed during audit")
    atomic_write_json(
        args.output,
        {
            "status": "passed" if not failed_count else "row_contract_errors",
            "scope": "full normalized corpus"
            if limit == corpus_rows
            else "prefix timing diagnostic",
            "rows": limit,
            "corpus_rows": corpus_rows,
            "data_sha256": before_sha,
            "manifest_sha256": sha256_file(args.manifest),
            "script_sha256": sha256_file(__file__),
            "scientific_id": content_id(scientific_payload(config), prefix="sci"),
            "max_length": config.model.language.max_length,
            "patches_per_image": patches,
            "seconds": time.monotonic() - started,
            "sources": dict(groups),
            "longest_examples_by_source": dict(longest),
            "row_contract_errors": failed_count,
            "first_errors": failures,
            "interpretation": "All-masked individual rows are retained context, not rejected rows. "
            "This row audit does not establish distributed microbatch viability. "
            "Counts are unique-row supervision, not sampled epoch exposure.",
        },
    )
    if failed_count:
        raise SystemExit(f"{failed_count} row contract errors; inspect {args.output}")


if __name__ == "__main__":
    main()
