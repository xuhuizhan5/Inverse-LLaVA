from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from invllava.config.schema import OptimizerSpec


def prepare_trainable_parameters(
    module: nn.Module,
    spec: OptimizerSpec,
    *,
    external_fp32_master: bool = False,
) -> dict[str, int | str]:
    """Apply the declared optimizer-storage precision before device placement.

    PyTorch AdamW creates moment tensors with the parameter dtype. Models in
    this project are loaded in BF16, so plain AdamW would otherwise update both
    trainable weights and moments in BF16. The manuscript recipe used
    DeepSpeed's FP32 master parameters and optimizer states. Keeping the small
    trainable partition in FP32 gives the portable single-process/DDP path the
    same high-precision update state while autocast still executes supported
    matrix operations in BF16.
    """

    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    if external_fp32_master and spec.update_dtype != "float32":
        raise ValueError("the DeepSpeed ZeRO-2 path requires update_dtype=float32")
    if spec.update_dtype == "float32" and not external_fp32_master:
        with torch.no_grad():
            for parameter in trainable:
                parameter.data = parameter.data.float()
    dtypes = {str(parameter.dtype) for parameter in trainable}
    if len(dtypes) != 1:
        raise RuntimeError(f"trainable parameters have mixed storage dtypes: {sorted(dtypes)}")
    return {
        "policy": spec.update_dtype,
        "implementation": (
            "deepspeed-fp32-master" if external_fp32_master else "parameter-storage"
        ),
        "dtype": next(iter(dtypes)),
        "parameters": sum(parameter.numel() for parameter in trainable),
        "tensors": len(trainable),
    }


def build_optimizer(
    parameters: Iterable[nn.Parameter], spec: OptimizerSpec
) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(
        trainable,
        lr=spec.learning_rate,
        betas=spec.betas,
        eps=spec.eps,
        weight_decay=spec.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    spec: OptimizerSpec,
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = round(total_steps * spec.warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if spec.scheduler == "constant":
            return 1.0
        if spec.scheduler == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
