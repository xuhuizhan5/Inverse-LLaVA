#!/usr/bin/env python3
"""Record bounded NCCL all-reduce latency and effective bandwidth.

Run with ``torchrun`` on one node. This is an admission diagnostic rather than
a model-performance benchmark; it catches unexpectedly slow peer transport
before a paid multi-GPU training run.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist

from invllava.artifacts.atomic import atomic_write_json


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item)
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be comma-separated positive MiB values")
    return sizes


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes-mib", type=_parse_sizes, default=(64, 256))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 2:
        raise ValueError("NCCL benchmark requires at least one warmup and two repetitions")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL benchmark requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("NCCL bandwidth qualification requires at least two ranks")

    results: list[dict[str, float | int | str]] = []
    for size_mib in args.sizes_mib:
        byte_count = size_mib * 1024 * 1024
        element_count = byte_count // torch.tensor([], dtype=torch.bfloat16).element_size()
        value = torch.ones(element_count, device=device, dtype=torch.bfloat16)
        for _ in range(args.warmups):
            dist.all_reduce(value)
        torch.cuda.synchronize(device)
        dist.barrier()

        elapsed: list[float] = []
        for _ in range(args.repetitions):
            dist.barrier()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            dist.all_reduce(value)
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - started)

        local = torch.tensor(
            [statistics.median(elapsed), _percentile(elapsed, 0.9)],
            device=device,
            dtype=torch.float64,
        )
        gathered = [torch.zeros_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        if rank == 0:
            rank_medians = [float(item[0].item()) for item in gathered]
            rank_p90 = [float(item[1].item()) for item in gathered]
            median_seconds = max(rank_medians)
            p90_seconds = max(rank_p90)
            algorithmic_gbps = byte_count / median_seconds / 1e9
            bus_factor = 2.0 * (world_size - 1) / world_size
            results.append(
                {
                    "size_mib": size_mib,
                    "dtype": "bfloat16",
                    "median_seconds_max_rank": median_seconds,
                    "p90_seconds_max_rank": p90_seconds,
                    "algorithmic_gbps": algorithmic_gbps,
                    "estimated_bus_gbps": algorithmic_gbps * bus_factor,
                }
            )
        del value

    if rank == 0:
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "backend": dist.get_backend(),
            "world_size": world_size,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(world_size)],
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "sizes_mib": list(args.sizes_mib),
            "nccl_environment": {
                key: value
                for key in (
                    "NCCL_DEBUG",
                    "NCCL_IB_DISABLE",
                    "NCCL_P2P_DISABLE",
                    "NCCL_SOCKET_IFNAME",
                )
                if (value := os.environ.get(key)) is not None
            },
            "results": results,
        }
        atomic_write_json(args.output, payload)
        print(payload)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
