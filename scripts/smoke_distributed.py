#!/usr/bin/env python3
"""Exercise one NCCL collective and record the visible multi-GPU topology."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("distributed CUDA smoke requires a visible GPU")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    try:
        value = torch.tensor(float(rank + 1), device=device)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2
        if value.item() != expected:
            raise RuntimeError(f"NCCL all-reduce returned {value.item()}, expected {expected}")

        identity = {
            "rank": rank,
            "local_rank": local_rank,
            "device": torch.cuda.get_device_name(local_rank),
            "uuid": str(torch.cuda.get_device_properties(local_rank).uuid),
            "capability": list(torch.cuda.get_device_capability(local_rank)),
        }
        identities: list[dict[str, object] | None] = [None] * world_size
        dist.all_gather_object(identities, identity)
        if rank == 0:
            payload = {
                "backend": dist.get_backend(),
                "world_size": world_size,
                "all_reduce_sum": value.item(),
                "devices": identities,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".partial")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
