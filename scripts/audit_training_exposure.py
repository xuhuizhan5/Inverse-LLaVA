#!/usr/bin/env python3
"""Audit one epoch's actual distributed row/target exposure without loading images.

Uses the training sampler and installed Accelerate batch sharding. Run on the
training host with cached tokenizers before a token-matched control. Final
training metrics must confirm these predictions; a unique-row budget alone
does not include the repeated rows used to equalize rank batch lengths.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from torch.utils.data import BatchSampler

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256
from invllava.config.identifiers import content_id, scientific_payload
from invllava.config.loader import ConfigRepository
from invllava.data.collate import encode_vicuna_v1
from invllava.data.controls import supervised_tokens_after_expansion
from invllava.data.dataset import NormalizedConversationDataset
from invllava.data.manifest import PreparedDatasetManifest
from invllava.data.sampling import stratified_nested_indices
from invllava.model.loaders import load_tokenizer
from invllava.train.sampler import ModalityLengthGroupedSampler


def distributed_epoch_exposure(lengths, targets, *, batch_size, processes, accumulation, seed):
    from accelerate.data_loader import BatchSamplerShard

    if len(lengths) != len(targets) or not lengths or any(value < 0 for value in targets):
        raise ValueError("lengths and nonnegative target counts must align and be nonempty")
    if min(batch_size, processes, accumulation) <= 0:
        raise ValueError("batch, process, and accumulation counts must be positive")
    sampler = ModalityLengthGroupedSampler(
        lengths, batch_size=batch_size, group_count=processes * accumulation, seed=seed
    )
    exposures = Counter()
    batches_by_rank = []
    for rank in range(processes):
        batches = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
        # Accelerate leaves a single-process map-style loader unsharded, so its
        # last microbatch stays short. Only multiple processes pad equal ranks.
        if processes > 1:
            batches = BatchSamplerShard(
                batches,
                num_processes=processes,
                process_index=rank,
                split_batches=False,
                even_batches=True,
            )
        batches = list(batches)
        batches_by_rank.append(len(batches))
        exposures.update(index for batch in batches for index in batch)
    if len(set(batches_by_rank)) != 1 or set(exposures) != set(range(len(lengths))):
        raise RuntimeError("distributed epoch did not cover all rows with equal rank steps")
    return {
        "unique_rows": len(lengths),
        "sampled_rows": sum(exposures.values()),
        "repeated_rows": sum(exposures.values()) - len(lengths),
        "unique_supervised_tokens": sum(targets),
        "sampled_supervised_tokens": sum(
            targets[index] * count for index, count in exposures.items()
        ),
        "microbatches_per_rank": batches_by_rank[0],
        "optimizer_updates": math.ceil(batches_by_rank[0] / accumulation),
        "exposure_counts": [exposures[index] for index in range(len(lengths))],
    }


def main():
    import accelerate
    from accelerate.utils import DataLoaderConfiguration
    from transformers import CLIPVisionConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        nargs=3,
        action="append",
        required=True,
        metavar=("EXPERIMENT", "JSONL", "MANIFEST"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    defaults = DataLoaderConfiguration()
    if defaults.split_batches or not defaults.even_batches:
        raise RuntimeError("Accelerate default batching changed; review the training contract")
    datasets = {}
    repository = ConfigRepository()
    records = []
    tokenizer = None
    reference_model = None
    for experiment, jsonl, manifest_path in args.condition:
        config = repository.resolve(experiment)
        if config.training.epochs != 1 or not config.training.group_by_modality_length:
            raise ValueError("this audit covers the reviewed one-epoch grouped-sampler recipe")
        if tokenizer is None:
            reference_model = config.model
            tokenizer = load_tokenizer(config.model, local_files_only=True)
            vision = CLIPVisionConfig.from_pretrained(
                config.model.vision.checkpoint,
                revision=config.model.vision.revision,
                local_files_only=True,
            )
            patches = (config.model.vision.image_size // vision.patch_size) ** 2
            if config.model.vision.feature_select == "cls_patch":
                patches += 1
        elif config.model != reference_model:
            raise ValueError("conditions must use the same model and tokenization contract")
        manifest = PreparedDatasetManifest.read(manifest_path)
        manifest.verify(
            jsonl,
            expected_data_id=config.data.id,
            expected_source_revision=config.data.annotation.revision,
            expected_source_sha256=config.data.annotation.sha256,
            expected_source_filter=config.data.include_sources,
        )
        if jsonl not in datasets:
            datasets[jsonl] = NormalizedConversationDataset(jsonl)
        dataset = datasets[jsonl]
        indices = stratified_nested_indices(
            dataset.ids,
            dataset.sources,
            fraction=config.data.sample_fraction,
            seed=config.data.sample_seed,
            maximum=config.data.max_samples,
        )
        targets = []
        for index in indices:
            sample = dataset[index]
            input_ids, labels = encode_vicuna_v1(tokenizer, sample.turns)
            targets.append(
                supervised_tokens_after_expansion(
                    input_ids,
                    labels,
                    image_patch_counts=[patches] * len(sample.images),
                    max_length=config.model.language.max_length,
                )
            )
        exposure = distributed_epoch_exposure(
            [dataset.modality_lengths[index] for index in indices],
            targets,
            batch_size=config.training.per_device_batch_size,
            processes=config.runtime.num_processes,
            accumulation=config.training.gradient_accumulation_steps,
            seed=config.training.seed,
        )
        records.append(
            {
                "experiment": config.id,
                "data_manifest_sha256": sha256_file(manifest_path),
                "scientific_id": content_id(scientific_payload(config), prefix="sci"),
                "resolved_config": config.model_dump(mode="json"),
                "selected_ids": [dataset.ids[index] for index in indices],
                "targets_per_unique_row": targets,
                **exposure,
            }
        )
        print(
            config.id,
            {key: value for key, value in exposure.items() if key != "exposure_counts"},
            flush=True,
        )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "epoch": 0,
            "accelerate_version": accelerate.__version__,
            "execution_source_sha256": execution_source_sha256(Path(__file__).resolve().parents[1])[
                0
            ],
            "sharding": "single process: unsharded; multiple processes: "
            "BatchSamplerShard with split_batches=False, even_batches=True",
            "image_loading": "none; prepared image integrity verified from manifest",
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
