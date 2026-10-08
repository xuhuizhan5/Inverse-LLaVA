#!/usr/bin/env python3
"""Create sealed token, mixture, and correspondence controls on one filesystem."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.config.loader import ConfigRepository
from invllava.data.collate import encode_vicuna_v1
from invllava.data.control_packages import materialize_control_package
from invllava.data.controls import (
    collect_supervised_prefix,
    deterministic_image_derangement,
    prefix_ids,
    supervised_tokens_after_expansion,
    token_matched_prefix,
)
from invllava.data.dataset import NormalizedConversationDataset
from invllava.data.manifest import PreparedDatasetManifest
from invllava.data.sampling import stratified_nested_indices
from invllava.data.types import ConversationSample

CONTROL_REVISION = "continuation-controls-v2-expanded-targets"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-jsonl", required=True)
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--instruction-jsonl", required=True)
    parser.add_argument("--instruction-manifest", required=True)
    parser.add_argument("--experiment", default="configs/experiment/paired_continuation_base.yaml")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows", type=int, default=5580)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def _load_verified(jsonl: str, manifest_path: str) -> NormalizedConversationDataset:
    manifest = PreparedDatasetManifest.read(manifest_path)
    manifest.verify(
        jsonl,
        expected_data_id=manifest.data_id,
        expected_source_revision=manifest.source_revision,
        expected_source_filter=tuple(manifest.source_filter or ()),
    )
    return NormalizedConversationDataset(jsonl)


def _ordered_subset(
    dataset: NormalizedConversationDataset,
    *,
    rows: int,
    seed: int,
) -> list[ConversationSample]:
    indices = stratified_nested_indices(
        dataset.ids,
        dataset.sources,
        fraction=1.0,
        seed=seed,
        maximum=rows,
    )
    if len(indices) != rows:
        raise ValueError(f"requested {rows} rows from a dataset containing {len(dataset)}")
    return [dataset[index] for index in indices]


def _token_counts(
    samples: list[ConversationSample],
    tokenizer: object,
    max_length: int,
    *,
    patches_per_image: int,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        input_ids, labels = encode_vicuna_v1(tokenizer, sample.turns)
        count = supervised_tokens_after_expansion(
            input_ids,
            labels,
            image_patch_counts=[patches_per_image] * len(sample.images),
            max_length=max_length,
        )
        if sample.id in counts:
            raise ValueError(f"duplicate control sample identity: {sample.id}")
        counts[sample.id] = count
    return counts


def main() -> None:
    args = _arguments()
    if args.rows < 2 or args.workers <= 0:
        raise ValueError("--rows must be at least two and --workers must be positive")
    downloads_authorized = args.allow_download and os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") == "1"
    if args.allow_download and not downloads_authorized:
        raise PermissionError("--allow-download also requires INVLLAVA_ALLOW_DOWNLOADS=1")
    if not downloads_authorized:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    paired_manifest = PreparedDatasetManifest.read(args.paired_manifest)
    instruction_manifest = PreparedDatasetManifest.read(args.instruction_manifest)
    paired = _load_verified(args.paired_jsonl, args.paired_manifest)
    instruction = _load_verified(args.instruction_jsonl, args.instruction_manifest)
    paired_rows = _ordered_subset(paired, rows=args.rows, seed=args.seed)
    instruction_indices = stratified_nested_indices(
        instruction.ids, instruction.sources, fraction=1.0, seed=args.seed
    )

    repository = ConfigRepository(args.config_root)
    resolved = repository.resolve(args.experiment)
    from invllava.model.loaders import load_tokenizer

    tokenizer = load_tokenizer(resolved.model, local_files_only=not downloads_authorized)
    from transformers import AutoConfig

    vision_config = AutoConfig.from_pretrained(
        resolved.model.vision.checkpoint,
        revision=resolved.model.vision.revision,
        local_files_only=not downloads_authorized,
    )
    vision_config = getattr(vision_config, "vision_config", vision_config)
    patch_size = int(vision_config.patch_size)
    if resolved.model.vision.image_size % patch_size:
        raise ValueError("vision image size is not divisible by patch size")
    patches = (resolved.model.vision.image_size // patch_size) ** 2
    patches += int(resolved.model.vision.feature_select == "cls_patch")
    paired_counts = _token_counts(
        paired_rows, tokenizer, resolved.model.language.max_length, patches_per_image=patches
    )
    if any(count <= 0 for count in paired_counts.values()):
        raise ValueError("paired treatment contains a row with no surviving target")
    paired_tokens = sum(paired_counts.values())
    print(f"Counted {len(paired_rows)} paired rows: {paired_tokens} surviving targets.", flush=True)

    def instruction_tokens(sample: ConversationSample) -> int:
        return _token_counts(
            [sample], tokenizer, resolved.model.language.max_length, patches_per_image=patches
        )[sample.id]

    prefix = collect_supervised_prefix(
        (instruction[index] for index in instruction_indices),
        instruction_tokens,
        minimum_rows=args.rows,
        target_tokens=paired_tokens,
    )
    instruction_order = list(prefix.samples)
    instruction_counts = prefix.token_counts
    print(f"Audited {prefix.examined_rows} instruction rows for the required controls.", flush=True)
    counts = {
        **{f"paired:{key}": value for key, value in paired_counts.items()},
        **{f"shuffled:{key}": value for key, value in paired_counts.items()},
        **{f"instruction:{key}": value for key, value in instruction_counts.items()},
    }
    token_control = token_matched_prefix(
        instruction_order,
        instruction_counts,
        target_tokens=paired_tokens,
    )
    half = args.rows // 2
    mixed = [
        *prefix_ids(paired_rows[:half], "paired"),
        *prefix_ids(instruction_order[: args.rows - half], "instruction"),
    ]
    shuffled = prefix_ids(
        deterministic_image_derangement(paired_rows, seed=args.seed),
        "shuffled",
    )
    parents = {
        "paired_manifest_sha256": sha256_file(args.paired_manifest),
        "instruction_manifest_sha256": sha256_file(args.instruction_manifest),
        "paired_normalized_sha256": paired_manifest.normalized_sha256,
        "instruction_normalized_sha256": instruction_manifest.normalized_sha256,
    }
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    atomic_write_json(
        output_root / "supervision-audit.json",
        {
            "algorithm": CONTROL_REVISION,
            "max_length": resolved.model.language.max_length,
            "patches_per_image": patches,
            "paired_rows": len(paired_rows),
            "paired_supervised_tokens": paired_tokens,
            "instruction_pool_rows": len(instruction),
            "instruction_rows_examined": prefix.examined_rows,
            "instruction_rows_without_surviving_targets": list(prefix.excluded_ids),
            "audit_scope": "deterministic prefix sufficient for row and token controls",
            "selection_policy": "exclude zero-target rows from instruction control eligibility",
        },
    )
    reports = [
        materialize_control_package(
            output_root,
            revision=CONTROL_REVISION,
            name="instruction-token-matched-1pct",
            data_id="instruction-token-matched-1pct",
            samples=prefix_ids(token_control, "instruction"),
            parents=parents,
            token_counts=counts,
            workers=args.workers,
        ),
        materialize_control_package(
            output_root,
            revision=CONTROL_REVISION,
            name="paired-instruction-mixed-1pct",
            data_id="paired-instruction-mixed-1pct",
            samples=mixed,
            parents=parents,
            token_counts=counts,
            workers=args.workers,
        ),
        materialize_control_package(
            output_root,
            revision=CONTROL_REVISION,
            name="paired-shuffled-1pct",
            data_id="paired-shuffled-1pct",
            samples=shuffled,
            parents=parents,
            token_counts=counts,
            workers=args.workers,
        ),
    ]
    atomic_write_json(output_root / "continuation-controls.manifest.json", reports)
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
