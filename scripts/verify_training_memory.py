#!/usr/bin/env python3
"""Exercise native inverse and projector models with full-length batches.

This measures allocation feasibility, including optimizer state. Synthetic
losses, throughput, and resulting weights are not scientific training results.
Launch with torch.distributed.run using the recipe's process count. Keep an
external timeout and GPU shutdown guard around the command.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import invllava
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.artifacts.source import execution_source_sha256
from invllava.config.loader import ConfigRepository
from invllava.config.validation import require_frozen_execution


class FullLengthWorkload(Dataset):
    """Maximum-length rows, followed by a minimally padded attention workload."""

    def __init__(
        self,
        *,
        rows: int,
        length: int,
        patches: int,
        image_size: int,
        dense_rows: int | None = None,
    ):
        if rows <= 0 or patches < 0 or length - patches < 2 or image_size <= 0:
            raise ValueError("invalid full-length memory workload")
        self.rows, self.length, self.patches, self.image_size = rows, length, patches, image_size
        if dense_rows is not None and dense_rows <= 0:
            raise ValueError("dense prefix must contain a positive number of rows")
        self.dense_rows = dense_rows

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        if not 0 <= index < self.rows:
            raise IndexError(index)
        return index

    def collate(self, indices):
        count = len(indices)
        tokens = self.length - self.patches + int(self.patches > 0)
        ids = torch.full((count, tokens), 3, dtype=torch.long)
        ids[:, 0] = 1
        labels = ids.clone()
        labels[:, 0] = -100
        images = [[] for _ in indices]
        if self.patches:
            ids[:, 1] = -200
            labels[:, 1] = -100
            images = [[torch.zeros(3, self.image_size, self.image_size)] for _ in indices]
        attention = torch.ones_like(ids, dtype=torch.bool)
        for row, index in enumerate(indices):
            if (
                self.dense_rows is not None
                and index >= self.dense_rows
                and (index % self.dense_rows) % 2
            ):
                ids[row, -1], labels[row, -1], attention[row, -1] = 0, -100, False
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attention,
            "pixel_values": images,
            "sample_ids": [f"synthetic-memory-{i}" for i in indices],
        }


def verify_history(history, *, updates, global_batch, length, padded_tokens_per_update=0):
    if [row["step"] for row in history] != list(range(1, updates + 1)):
        raise ValueError("memory probe update history is incomplete")
    for row in history:
        if row["train/samples_seen"] != row["step"] * global_batch:
            raise ValueError("memory probe sample exposure differs from its contract")
        expected_tokens = (
            row["step"] * global_batch * length - max(0, row["step"] - 1) * padded_tokens_per_update
        )
        if row["train/tokens_seen"] != expected_tokens:
            raise ValueError("memory probe did not exercise the declared expanded length")
        if not all(math.isfinite(row[key]) for key in ("train/loss", "train/gradient_norm")):
            raise ValueError("memory probe recorded nonfinite optimization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--initial-checkpoint", required=True, type=Path)
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--microbatch-size", type=int)
    parser.add_argument("--modality", choices=("image", "text"), default="image")
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not 2 <= args.updates <= 4:
        parser.error("use two to four optimizer updates for this bounded diagnostic")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = ConfigRepository(args.config_root).resolve(
        args.experiment, microbatch_size=args.microbatch_size
    )
    require_frozen_execution(config)
    if (
        config.runtime.accelerator != "cuda"
        or config.runtime.distributed_strategy != "deepspeed_zero2"
    ):
        raise ValueError("this diagnostic requires the reviewed CUDA ZeRO-2 recipe")
    if int(os.environ.get("WORLD_SIZE", "1")) != config.runtime.num_processes:
        raise ValueError("launch process count differs from the recipe")
    if os.environ.get("PYTORCH_ALLOC_CONF") != config.runtime.cuda_allocator_conf:
        raise ValueError("set PYTORCH_ALLOC_CONF to the resolved recipe before launching Python")
    digest = sha256_file(args.initial_checkpoint / "model_delta.safetensors")
    if config.initial_checkpoint_id != f"sha256:{digest}":
        raise ValueError("initial checkpoint differs from the experiment")
    source = Path(invllava.__file__).resolve().parents[2]
    source_sha = execution_source_sha256(source)[0]
    expected_source = os.environ.get("INVLLAVA_EXECUTION_SOURCE_SHA256")
    if expected_source is not None and expected_source != source_sha:
        raise ValueError("memory probe source differs from the expected imported checkout")
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    from invllava.model.loaders import build_model
    from invllava.runtime.cache import configure_runtime_cache
    from invllava.runtime.numerics import configure_torch_numerics
    from invllava.runtime.optimization import configure_model_kernels
    from invllava.train.checkpoint import load_trainable_weights
    from invllava.train.engine import TrainingEngine
    from invllava.train.state import seed_everything

    configure_runtime_cache(config.runtime.cache_root)
    configure_torch_numerics(config.runtime)
    seed_everything(config.training.seed)
    model, _ = build_model(
        config.model, attention_backend=config.runtime.attention_backend, local_files_only=True
    )
    load_trainable_weights(args.initial_checkpoint, model)
    kernel_report = configure_model_kernels(model, config.runtime.kernel_optimization)
    vision = model.vision_encoder.tower.config
    patches = (vision.image_size // vision.patch_size) ** 2
    if config.model.vision.feature_select != "patch":
        raise ValueError("memory workload requires the declared patch-only vision interface")
    batch = config.training.per_device_batch_size
    global_batch = (
        batch * config.training.gradient_accumulation_steps * config.runtime.num_processes
    )
    workload = FullLengthWorkload(
        rows=global_batch * args.updates,
        length=config.model.language.max_length,
        patches=patches if args.modality == "image" else 0,
        image_size=config.model.vision.image_size,
        dense_rows=global_batch,
    )
    # Keep model, optimizer, precision, accumulation, checkpointing and engine
    # unchanged. Only the synthetic workload and short diagnostic schedule differ.
    payload = config.model_dump(mode="json")
    payload["training"].update(
        epochs=1, log_every_steps=1, save_every_steps=args.updates, checkpoint_milestones=[1.0]
    )
    config = type(config).model_validate(payload)
    loader = DataLoader(
        workload, batch_size=batch, shuffle=False, num_workers=0, collate_fn=workload.collate
    )
    engine = TrainingEngine(config=config, model=model, dataloader=loader, run_dir=args.output_dir)
    if engine.accelerator.is_main_process:
        atomic_write_json(args.output_dir / "diagnostic-config.json", payload)
    try:
        engine.run()
        if engine.accelerator.is_main_process:
            metrics = args.output_dir / "metrics.jsonl"
            history = [json.loads(line) for line in metrics.read_text().splitlines()]
            verify_history(
                history,
                updates=args.updates,
                global_batch=global_batch,
                length=workload.length,
                padded_tokens_per_update=global_batch // 2,
            )
            atomic_write_json(
                args.output_dir / "acceptance.json",
                {
                    "status": "passed",
                    "kind": "synthetic-training-memory",
                    "model_architecture": config.model.architecture,
                    "execution_source_sha256": source_sha,
                    "kernel_optimization": kernel_report.to_dict(),
                    "verifier_sha256": sha256_file(__file__),
                    "initialization_sha256": digest,
                    "metrics_sha256": sha256_file(metrics),
                    "modality": args.modality,
                    "expanded_length": workload.length,
                    "padding_policy": (
                        "dense-first-update; subsequent updates have one trailing pad "
                        "in alternating rows"
                    ),
                    "global_batch": global_batch,
                    "microbatch": batch,
                    "updates": args.updates,
                    "peak_allocated_bytes": max(
                        row["train/peak_allocated_bytes"] for row in history
                    ),
                    "peak_reserved_bytes": max(row["train/peak_reserved_bytes"] for row in history),
                    "scope": (
                        "Synthetic full-length operational memory test. No benchmark accuracy, "
                        "data throughput, resume parity, or model-quality claim. Generated weights "
                        "must not be released as a trained research model."
                    ),
                },
            )
        engine.accelerator.wait_for_everyone()
    except BaseException as error:
        atomic_write_json(
            args.output_dir / f"failure-rank-{engine.accelerator.process_index}.json",
            {"status": "failed", "error_type": type(error).__name__, "reason": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
