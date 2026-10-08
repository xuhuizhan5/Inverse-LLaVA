"""Minimal LoRA implementation with an auditable target-selection policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: int, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        initialization_factory = {"device": base.weight.device, "dtype": torch.float32}
        self.lora_a = nn.Linear(base.in_features, rank, bias=False, **initialization_factory)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False, **initialization_factory)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(dtype=base.weight.dtype)
        self.lora_b.to(dtype=base.weight.dtype)

    def forward(self, value: Tensor) -> Tensor:
        return self.base(value) + self.lora_b(self.lora_a(self.dropout(value))) * self.scale


def _parent_and_attribute(module: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if isinstance(parent, (nn.ModuleList, nn.Sequential))
            else getattr(parent, part)
        )
    return parent, parts[-1]


def apply_lora(
    module: nn.Module,
    *,
    target_suffixes: tuple[str, ...],
    rank: int,
    alpha: int,
    dropout: float,
    excluded_layer_indices: tuple[int, ...] = (),
) -> tuple[str, ...]:
    """Replace selected linears once; return exact fully-qualified targets."""

    selected: list[str] = []
    for name, child in tuple(module.named_modules()):
        if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        if ".fusion." in name:
            continue
        if any(f".layers.{index}." in f".{name}." for index in excluded_layer_indices):
            continue
        parent, attribute = _parent_and_attribute(module, name)
        setattr(parent, attribute, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        selected.append(name)
    if not selected:
        raise ValueError("LoRA policy selected no linear modules")
    return tuple(selected)


def set_lora_trainable(module: nn.Module, trainable: bool) -> tuple[str, ...]:
    """Set only adapter tensors trainable while leaving wrapped bases frozen."""

    selected: list[str] = []
    for name, child in module.named_modules():
        if not isinstance(child, LoRALinear):
            continue
        child.base.requires_grad_(False)
        child.lora_a.requires_grad_(trainable)
        child.lora_b.requires_grad_(trainable)
        selected.append(name)
    if not selected:
        raise ValueError("model contains no LoRA modules")
    return tuple(selected)


@torch.no_grad()
def merge_lora_for_inference(module: nn.Module) -> tuple[str, ...]:
    """Fold every LoRA delta into its base linear and remove adapter modules."""

    selected = [
        (name, child)
        for name, child in tuple(module.named_modules())
        if isinstance(child, LoRALinear)
    ]
    for name, adapter in selected:
        base = adapter.base
        merged = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        delta = adapter.lora_b.weight @ adapter.lora_a.weight
        merged.weight.copy_(base.weight + delta * adapter.scale)
        if base.bias is not None:
            assert merged.bias is not None
            merged.bias.copy_(base.bias)
        merged.requires_grad_(False)
        merged.train(module.training)
        parent, attribute = _parent_and_attribute(module, name)
        setattr(parent, attribute, merged)
    return tuple(name for name, _ in selected)


@dataclass(frozen=True)
class ParameterAudit:
    total: int
    trainable: int
    trainable_names: tuple[str, ...]

    @property
    def fraction(self) -> float:
        return self.trainable / self.total if self.total else 0.0


def audit_parameters(module: nn.Module) -> ParameterAudit:
    trainable_names = tuple(
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    )
    return ParameterAudit(
        total=sum(parameter.numel() for parameter in module.parameters()),
        trainable=sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
        trainable_names=trainable_names,
    )
