from __future__ import annotations

import torch
from torch import nn

from invllava.config.schema import OptimizerSpec
from invllava.train.optimizer import build_optimizer, prepare_trainable_parameters


class _MixedPrecisionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.trainable = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.frozen.requires_grad_(False)


def test_fp32_trainable_policy_preserves_frozen_model_dtype() -> None:
    model = _MixedPrecisionModule()

    summary = prepare_trainable_parameters(model, OptimizerSpec())

    assert model.frozen.weight.dtype == torch.bfloat16
    assert model.trainable.weight.dtype == torch.float32
    assert summary == {
        "policy": "float32",
        "implementation": "parameter-storage",
        "dtype": "torch.float32",
        "parameters": 16,
        "tensors": 1,
    }


def test_fp32_trainable_policy_creates_fp32_adam_moments() -> None:
    model = _MixedPrecisionModule()
    prepare_trainable_parameters(model, OptimizerSpec())
    optimizer = build_optimizer(model.parameters(), OptimizerSpec())
    model.trainable.weight.grad = torch.ones_like(model.trainable.weight)

    optimizer.step()

    state = optimizer.state[model.trainable.weight]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_model_dtype_policy_remains_available_for_diagnostics() -> None:
    model = _MixedPrecisionModule()
    spec = OptimizerSpec(update_dtype="model")

    summary = prepare_trainable_parameters(model, spec)

    assert model.trainable.weight.dtype == torch.bfloat16
    assert summary["dtype"] == "torch.bfloat16"


def test_external_fp32_master_preserves_bf16_forward_parameters() -> None:
    model = _MixedPrecisionModule()

    summary = prepare_trainable_parameters(model, OptimizerSpec(), external_fp32_master=True)

    assert model.trainable.weight.dtype == torch.bfloat16
    assert summary["implementation"] == "deepspeed-fp32-master"
