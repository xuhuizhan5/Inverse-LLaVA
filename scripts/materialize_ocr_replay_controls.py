#!/usr/bin/env python3
"""Prepare OCR replay and image-bearing non-OCR controls without eval outcomes."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.loader import ConfigRepository
from invllava.data.collate import encode_vicuna_v1
from invllava.data.control_packages import materialize_control_package
from invllava.data.controls import (
    collect_supervised_prefix,
    prefix_ids,
    supervised_tokens_after_expansion,
    token_matched_prefix,
)
from invllava.data.dataset import NormalizedConversationDataset
from invllava.data.manifest import PreparedDatasetManifest
from invllava.data.sampling import stratified_nested_indices
from invllava.data.types import ConversationSample

REVISION = "ocr-replay-controls-v1-expanded-targets"
OCR_SOURCES = frozenset({"ocr_vqa", "textvqa"})
# Source tags follow image folders: textvqa here supplies TextCaps captions,
# while ocr_vqa supplies book-oriented QA. Preserve both original objectives.
NON_OCR_SOURCES = frozenset({"coco", "gqa", "vg"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-jsonl", required=True)
    parser.add_argument("--instruction-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment", default="configs/experiment/paired_continuation_base.yaml")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--rows", type=int, default=5580)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.rows <= 0 or args.workers <= 0:
        raise ValueError("row and worker counts must be positive")
    root = Path(args.output_root)
    if root.exists():
        raise FileExistsError(root)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = PreparedDatasetManifest.read(args.instruction_manifest)
    manifest.verify(
        args.instruction_jsonl,
        expected_data_id=manifest.data_id,
        expected_source_revision=manifest.source_revision,
        expected_source_filter=tuple(manifest.source_filter or ()),
    )
    dataset = NormalizedConversationDataset(args.instruction_jsonl)
    resolved = ConfigRepository(args.config_root).resolve(args.experiment)
    from transformers import AutoConfig

    from invllava.model.loaders import load_tokenizer

    tokenizer = load_tokenizer(resolved.model, local_files_only=True)
    vision = AutoConfig.from_pretrained(
        resolved.model.vision.checkpoint,
        revision=resolved.model.vision.revision,
        local_files_only=True,
    )
    vision = getattr(vision, "vision_config", vision)
    patch_size = int(vision.patch_size)
    if resolved.model.vision.image_size % patch_size:
        raise ValueError("image width must be divisible by patch width")
    patches = (resolved.model.vision.image_size // patch_size) ** 2
    patches += int(resolved.model.vision.feature_select == "cls_patch")

    def count(sample: ConversationSample) -> int:
        ids, labels = encode_vicuna_v1(tokenizer, sample.turns)
        return supervised_tokens_after_expansion(
            ids,
            labels,
            image_patch_counts=[patches] * len(sample.images),
            max_length=resolved.model.language.max_length,
        )

    order = stratified_nested_indices(dataset.ids, dataset.sources, fraction=1.0, seed=args.seed)

    def eligible(sources: frozenset[str]):
        for index in order:
            if dataset.sources[index] in sources and dataset.modality_lengths[index] > 0:
                sample = dataset[index]
                if len(sample.images) != 1:
                    raise ValueError("replay controls require one image per row")
                yield sample

    ocr = collect_supervised_prefix(
        eligible(OCR_SOURCES), count, minimum_rows=args.rows, target_tokens=1
    )
    targets = sum(ocr.token_counts.values())
    print(f"OCR replay: {len(ocr.samples)} rows, {targets} surviving targets.", flush=True)
    non_ocr = collect_supervised_prefix(
        eligible(NON_OCR_SOURCES), count, minimum_rows=args.rows, target_tokens=targets
    )
    plans = (
        ("ocr-replay-1pct", list(ocr.samples), ocr.token_counts),
        ("non-ocr-replay-rows-1pct", list(non_ocr.samples[: args.rows]), non_ocr.token_counts),
        (
            "non-ocr-replay-tokens-1pct",
            token_matched_prefix(non_ocr.samples, non_ocr.token_counts, target_tokens=targets),
            non_ocr.token_counts,
        ),
    )
    parents = {"instruction_manifest_sha256": sha256_file(args.instruction_manifest)}
    reports = []
    for name, samples, counts in plans:
        prefixed = prefix_ids(samples, "replay")
        report = materialize_control_package(
            root,
            name=name,
            data_id=name,
            revision=REVISION,
            samples=prefixed,
            parents=parents,
            token_counts={f"replay:{key}": value for key, value in counts.items()},
            workers=args.workers,
        )
        report["sources"] = dict(Counter(sample.source for sample in samples))
        reports.append(report)
        print(report, flush=True)
    atomic_write_json(
        root / "replay-controls.manifest.json",
        {
            "revision": REVISION,
            "seed": args.seed,
            "rows_requested": args.rows,
            "max_length": resolved.model.language.max_length,
            "patches_per_image": patches,
            "ocr_sources": sorted(OCR_SOURCES),
            "non_ocr_sources": sorted(NON_OCR_SOURCES),
            "eligibility": "one image, positive surviving causal targets; no evaluation inputs",
            "ocr_rows_examined": ocr.examined_rows,
            "non_ocr_rows_examined": non_ocr.examined_rows,
            "excluded_zero_target_ids": [*ocr.excluded_ids, *non_ocr.excluded_ids],
            "conditions": reports,
        },
    )


if __name__ == "__main__":
    main()
