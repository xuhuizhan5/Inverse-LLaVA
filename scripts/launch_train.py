#!/usr/bin/env python3
"""Launch exactly the process topology declared by an experiment config."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from invllava.artifacts.source import execution_source_sha256
from invllava.config.loader import ConfigRepository


def _apply_cuda_allocator_conf(environment: dict[str, str], expected: str | None) -> dict[str, str]:
    """Apply the resolved allocator policy before the worker imports PyTorch."""

    actual = environment.get("PYTORCH_ALLOC_CONF")
    if actual is not None and actual != expected:
        raise ValueError(
            "PYTORCH_ALLOC_CONF conflicts with the resolved runtime: "
            f"environment={actual!r}, runtime={expected!r}"
        )
    if expected is None:
        environment.pop("PYTORCH_ALLOC_CONF", None)
    else:
        environment["PYTORCH_ALLOC_CONF"] = expected
    return environment


def _apply_execution_source_identity(environment: dict[str, str], expected: str) -> dict[str, str]:
    """Bind every worker manifest to the source tree used by the launcher."""

    actual = environment.get("INVLLAVA_EXECUTION_SOURCE_SHA256")
    if actual is not None and actual != expected:
        raise ValueError(
            "INVLLAVA_EXECUTION_SOURCE_SHA256 conflicts with the launcher source: "
            f"environment={actual!r}, launcher={expected!r}"
        )
    environment["INVLLAVA_EXECUTION_SOURCE_SHA256"] = expected
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment")
    parser.add_argument("prepared_jsonl")
    parser.add_argument("prepared_manifest")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument(
        "--runtime-ref",
        help="execution-only runtime override; accumulation is derived to preserve global batch",
    )
    parser.add_argument(
        "--microbatch-size",
        type=int,
        help=(
            "execution microbatch per process; accumulation is derived to preserve "
            "the declared global optimizer-update batch"
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("on", "off"),
        help="recorded execution override used for hardware qualification",
    )
    parser.add_argument(
        "--maximum-samples",
        type=int,
        help="bounded execution subset recorded in the resolved scientific configuration",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-root")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--projector-checkpoint")
    parser.add_argument("--resume-from")
    parser.add_argument(
        "--resume-source-change-reason",
        help="required audit note when an exact resume uses a different execution source",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "permit pinned model acquisition during an explicit staging run; "
            "scientific runs should use the pre-populated offline cache"
        ),
    )
    args = parser.parse_args()

    if args.allow_download and os.environ.get("INVLLAVA_ALLOW_DOWNLOADS") != "1":
        raise PermissionError(
            "--allow-download also requires INVLLAVA_ALLOW_DOWNLOADS=1 on the staging host"
        )

    resolved = ConfigRepository(args.config_root).resolve(
        args.experiment,
        runtime_ref=args.runtime_ref,
        microbatch_size=args.microbatch_size,
        gradient_checkpointing=(
            args.gradient_checkpointing == "on" if args.gradient_checkpointing is not None else None
        ),
        maximum_samples=args.maximum_samples,
    )
    processes = resolved.runtime.num_processes
    if args.resume_from and not args.run_id:
        resume_path = Path(args.resume_from).resolve()
        if resume_path.parent.name != "checkpoints":
            raise ValueError("resume checkpoint must be <run>/checkpoints/<step>")
        run_id = resume_path.parent.parent.name
    else:
        run_id = args.run_id or (
            f"{resolved.id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
    global_batch = (
        processes
        * resolved.training.per_device_batch_size
        * resolved.training.gradient_accumulation_steps
    )
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        resolved.runtime.mixed_precision,
        "--dynamo_backend",
        "no",
    ]
    if processes > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            "-m",
            "invllava",
            "train",
            str(Path(args.experiment)),
            "--prepared-jsonl",
            str(Path(args.prepared_jsonl)),
            "--prepared-manifest",
            str(Path(args.prepared_manifest)),
            "--config-root",
            args.config_root,
            "--run-id",
            run_id,
        ]
    )
    if args.allow_download:
        command.append("--allow-download")
    if args.runtime_ref:
        command.extend(("--runtime-ref", args.runtime_ref))
    if args.microbatch_size is not None:
        command.extend(("--microbatch-size", str(args.microbatch_size)))
    if args.gradient_checkpointing is not None:
        command.extend(("--gradient-checkpointing", args.gradient_checkpointing))
    if args.maximum_samples is not None:
        command.extend(("--maximum-samples", str(args.maximum_samples)))
    if args.run_root:
        command.extend(("--run-root", args.run_root))
    for option, value in (
        ("--initial-checkpoint", args.initial_checkpoint),
        ("--projector-checkpoint", args.projector_checkpoint),
        ("--resume-from", args.resume_from),
        ("--resume-source-change-reason", args.resume_source_change_reason),
    ):
        if value:
            command.extend((option, value))
    print(
        (
            f"Launching {resolved.id}; run_id={run_id}; "
            f"batch={processes}x{resolved.training.per_device_batch_size}"
            f"x{resolved.training.gradient_accumulation_steps}={global_batch}; "
            f"gradient_checkpointing={resolved.training.gradient_checkpointing}; "
            f"strategy={resolved.runtime.distributed_strategy}"
        ),
        flush=True,
    )
    launch_environment = _apply_cuda_allocator_conf(
        os.environ.copy(), resolved.runtime.cuda_allocator_conf
    )
    repository_root = Path(__file__).resolve().parents[1]
    source_digest, _, _ = execution_source_sha256(repository_root)
    launch_environment = _apply_execution_source_identity(launch_environment, source_digest)
    if resolved.runtime.distributed_strategy == "deepspeed_zero2" and processes == 1:
        # Accelerate's single-process launcher does not publish rendezvous
        # variables. DeepSpeed still requires the standard distributed
        # environment for its one-rank correctness and checkpoint canaries.
        launch_environment.setdefault("RANK", "0")
        launch_environment.setdefault("LOCAL_RANK", "0")
        launch_environment.setdefault("WORLD_SIZE", "1")
        launch_environment.setdefault("LOCAL_WORLD_SIZE", "1")
        launch_environment.setdefault("MASTER_ADDR", "127.0.0.1")
        launch_environment.setdefault("MASTER_PORT", "29500")
    subprocess.run(command, check=True, env=launch_environment)


if __name__ == "__main__":
    main()
