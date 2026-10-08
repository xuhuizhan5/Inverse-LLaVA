#!/usr/bin/env python3
"""Fresh-process, offline acceptance of a sealed release on real benchmark inputs.

This is a loading/numerical gate. Shortened generations and zeroed visual
features are diagnostics and must not be reported as benchmark scores.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.datasets import load_examples
from invllava.eval.types import generation_request
from invllava.model.modeling import LLaVAReferenceForConditionalGeneration
from invllava.model.types import ExpandedSequence
from invllava.release import load_pretrained


def attention_context(kernel: str):
    """Scope an optional PyTorch kernel choice to this diagnostic only."""
    if kernel == "auto":
        return nullcontext()
    backends = {"flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}
    if kernel not in backends:
        raise ValueError(f"unsupported diagnostic SDPA kernel: {kernel}")
    return sdpa_kernel(backends[kernel])


def write_failure(target: Path, *, checkpoint_sha256: str, reason: str, details: dict) -> None:
    """Preserve diagnostic evidence before a guarded worker exits."""
    if target.exists():
        raise FileExistsError(target)
    atomic_write_json(
        target,
        {
            "schema_version": 1,
            "status": "failed",
            "kind": "release-inference-acceptance",
            "checkpoint_sha256": checkpoint_sha256,
            "verifier_sha256": sha256_file(__file__),
            "reason": reason,
            "details": details,
            "scope": "Failed operational acceptance; checkpoint must not be promoted.",
        },
    )


def without_visual_input(sequence: ExpandedSequence, *, projected_tokens: bool) -> ExpandedSequence:
    """Ablate the interface's visual input while preserving text and sequence positions."""
    state = replace(
        sequence.fusion_state,
        visual_features=torch.zeros_like(sequence.fusion_state.visual_features),
    )
    embeddings = sequence.inputs_embeds
    if projected_tokens:
        embeddings = embeddings.masked_fill(state.vision_mask.unsqueeze(-1), 0)
    return replace(sequence, inputs_embeds=embeddings, fusion_state=state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--release")
    source.add_argument(
        "--experiment", help="Native experiment for an inverse or projector checkpoint"
    )
    parser.add_argument("--checkpoint", help="Required with --experiment")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--runtime-ref")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--sdpa-kernel",
        choices=("auto", "flash", "math"),
        default="auto",
        help="Optional diagnostic-only kernel; no silent backend fallback.",
    )
    args = parser.parse_args()
    if bool(args.experiment) != bool(args.checkpoint) or (args.release and args.runtime_ref):
        parser.error("use --release, or --experiment with --checkpoint and optional --runtime-ref")
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    examples = load_examples(args.examples)[: args.samples]
    if len(examples) != args.samples:
        raise ValueError("insufficient acceptance examples")
    if args.release:
        release = load_pretrained(
            args.release,
            cache_dir=args.cache_dir,
            local_files_only=True,
            device="cuda",
            dtype="bfloat16",
            attention_backend="sdpa",
            generation_cache="kv",
            lora_execution="unmerged",
            max_new_tokens=16,
        )
        input_identity = {
            "release_metadata_sha256": sha256_file(release.release_root / "metadata.json")
        }
    else:
        from invllava.runtime.native import load_native_inference_runtime

        release = load_native_inference_runtime(
            experiment=args.experiment,
            checkpoint=args.checkpoint,
            config_root=args.config_root,
            runtime_ref=args.runtime_ref,
            device="cuda",
            max_new_tokens=16,
            require_portable=True,
            local_files_only=True,
        )
        input_identity = {
            "experiment_sha256": sha256_file(args.experiment),
            "architecture": release.resolved.model.architecture,
            "checkpoint_metadata_sha256": sha256_file(Path(args.checkpoint) / "metadata.json"),
        }
    records = []
    numerical_policy = {**release.numerical_policy, "diagnostic_sdpa_kernel": args.sdpa_kernel}
    with torch.inference_mode(), attention_context(args.sdpa_kernel):
        for example in examples:
            request = generation_request(example)
            expanded = release.generator.prepare_many((request,))

            def logits(sequence, state):
                return (
                    release.model.language_model(
                        inputs_embeds=sequence.inputs_embeds,
                        attention_mask=sequence.attention_mask,
                        position_ids=sequence.position_ids,
                        fusion_state=state,
                        use_cache=False,
                    )
                    .logits[:, -1]
                    .float()
                )

            ordinary = logits(expanded, expanded.fusion_state)
            ablated = without_visual_input(
                expanded,
                projected_tokens=isinstance(release.model, LLaVAReferenceForConditionalGeneration),
            )
            intervention = logits(ablated, ablated.fusion_state)
            if not torch.isfinite(ordinary).all() or not torch.isfinite(intervention).all():
                write_failure(
                    target,
                    checkpoint_sha256=release.checkpoint_sha256,
                    reason="non-finite logits",
                    details={"sample_id": example.id, "numerical_policy": numerical_policy},
                )
                raise RuntimeError(f"non-finite logits for {example.id}")
            first = release.generator(request)
            second = release.generator(request)
            if first != second:
                write_failure(
                    target,
                    checkpoint_sha256=release.checkpoint_sha256,
                    reason="greedy-repeat mismatch",
                    details={
                        "sample_id": example.id,
                        "first_prediction": first,
                        "second_prediction": second,
                        "examples_sha256": sha256_file(args.examples),
                        "numerical_policy": numerical_policy,
                        "torch_version": torch.__version__,
                        "device": torch.cuda.get_device_name(),
                        "active_training_modules": [
                            name
                            for name, module in release.model.named_modules()
                            if module.training
                        ],
                        "completed_samples": records,
                    },
                )
                raise RuntimeError(f"greedy-repeat mismatch for {example.id}")
            records.append(
                {
                    "sample_id": example.id,
                    "images_sha256": [sha256_file(path) for path in request.images],
                    "prediction": first,
                    "repeat_identical": True,
                    "expanded_shape": list(expanded.inputs_embeds.shape),
                    "vision_positions": int(expanded.fusion_state.vision_mask.sum()),
                    "finite_logits": True,
                    "zero_visual_features_max_abs_logit_change": float(
                        (ordinary - intervention).abs().max()
                    ),
                }
            )
    if not any(row["zero_visual_features_max_abs_logit_change"] > 0 for row in records):
        write_failure(
            target,
            checkpoint_sha256=release.checkpoint_sha256,
            reason="no observed visual-feature influence",
            details={"samples": records, "numerical_policy": numerical_policy},
        )
        raise RuntimeError("visual features have no observed effect on acceptance logits")
    atomic_write_json(
        target,
        {
            "schema_version": 1,
            "status": "passed",
            "kind": "release-inference-acceptance",
            "checkpoint_sha256": release.checkpoint_sha256,
            **input_identity,
            "examples_sha256": sha256_file(args.examples),
            "numerical_policy": numerical_policy,
            "verifier_sha256": sha256_file(__file__),
            "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(),
            "model_parameters": sum(p.numel() for p in release.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in release.model.parameters() if p.requires_grad
            ),
            "samples": records,
            "scope": "Offline loader, repeatability, and feature-path gate; no accuracy claim.",
        },
    )
    print(target, flush=True)


if __name__ == "__main__":
    main()
