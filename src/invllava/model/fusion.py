"""Equations (2)-(4): text-to-vision fusion inside attention projections."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from invllava.model.types import FusionState


class FusionBlock(nn.Module):
    """Produce additive Q/K/V updates from complementary modality streams.

    Concatenation computes

        alpha_P * W_concat^P [ W_t2v H ; V ]

    and the attention module adds this to its ordinary frozen/LoRA projection.
    Each target has its own text mapper, output map, visual RMS normalization,
    and scale. Modality masks place text and visual streams at complementary
    sequence positions; cross-position interaction occurs in causal attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        visual_size: int,
        target_sizes: dict[str, int],
        targets: Iterable[str] = ("q", "k", "v"),
        operator: str = "concat",
        mapper_rank: int | None = None,
        mapper_trainable: bool = True,
        visual_normalization: str = "rms",
        visual_norm_eps: float = 1e-6,
        scale_mode: str = "learned",
        initial_scale: float = 1.0,
        mapper_init_std: float | None = None,
        output_init_std: float = 1e-4,
        parameter_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.visual_size = visual_size
        self.operator = operator
        self.targets = tuple(targets)
        if operator not in {"concat", "add", "gated", "disabled"}:
            raise ValueError(f"unsupported fusion operator: {operator}")
        if not self.targets:
            raise ValueError("at least one fusion target is required")
        if visual_normalization not in {"rms", "none"}:
            raise ValueError("visual_normalization must be rms or none")
        if visual_norm_eps <= 0:
            raise ValueError("visual_norm_eps must be positive")

        self.mapper_rank = mapper_rank
        linear_kwargs = {"dtype": parameter_dtype} if parameter_dtype is not None else {}
        self.text_to_vision = nn.ModuleDict(
            {
                target: self._build_mapper(
                    hidden_size,
                    visual_size,
                    mapper_rank,
                    linear_kwargs=linear_kwargs,
                )
                for target in self.targets
            }
        )
        self.visual_norm = nn.ModuleDict(
            {
                target: VisualRMSNorm(
                    visual_size,
                    visual_norm_eps,
                    parameter_dtype=parameter_dtype,
                )
                if visual_normalization == "rms"
                else nn.Identity()
                for target in self.targets
            }
        )
        input_size = 2 * visual_size if operator == "concat" else visual_size
        self.output = nn.ModuleDict(
            {
                target: nn.Linear(
                    input_size,
                    target_sizes[target],
                    bias=False,
                    **linear_kwargs,
                )
                for target in self.targets
            }
        )
        self.gate = (
            nn.Linear(2 * visual_size, visual_size, bias=True, **linear_kwargs)
            if operator == "gated"
            else None
        )
        scales = {
            target: nn.Parameter(torch.tensor(float(initial_scale), dtype=parameter_dtype))
            for target in self.targets
        }
        self.scales = nn.ParameterDict(scales)
        if scale_mode == "fixed":
            for parameter in self.scales.values():
                parameter.requires_grad_(False)
        elif scale_mode != "learned":
            raise ValueError("scale_mode must be learned or fixed")

        mapper_std = mapper_init_std or 1.0 / math.sqrt(visual_size)
        for mapper in self.text_to_vision.values():
            if isinstance(mapper, nn.Sequential):
                nn.init.normal_(mapper[0].weight, mean=0.0, std=mapper_std)
                # Preserve the direct mapper's output-variance scale across
                # ranks: Var(B A x) = Var(A x) when std(B)=1/sqrt(rank).
                nn.init.normal_(
                    mapper[1].weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(mapper_rank),
                )
            else:
                nn.init.normal_(mapper.weight, mean=0.0, std=mapper_std)
        self.text_to_vision.requires_grad_(mapper_trainable)
        for projection in self.output.values():
            nn.init.normal_(projection.weight, mean=0.0, std=output_init_std)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _build_mapper(
        hidden_size: int,
        visual_size: int,
        rank: int | None,
        *,
        linear_kwargs: dict[str, object],
    ) -> nn.Module:
        if rank is None:
            return nn.Linear(hidden_size, visual_size, bias=False, **linear_kwargs)
        return nn.Sequential(
            nn.Linear(hidden_size, rank, bias=False, **linear_kwargs),
            nn.Linear(rank, visual_size, bias=False, **linear_kwargs),
        )

    def _joint(self, hidden_states: Tensor, state: FusionState, *, target: str) -> Tensor:
        batch, sequence, _ = hidden_states.shape
        state.validate(batch=batch, sequence=sequence, feature_dim=self.visual_size)
        text = self.text_to_vision[target](hidden_states)
        text = text * state.text_mask.unsqueeze(-1).to(text.dtype)
        visual = self.visual_norm[target](state.visual_features)
        visual = visual * state.vision_mask.unsqueeze(-1).to(text.dtype)
        if self.operator == "concat":
            return torch.cat((text, visual), dim=-1)
        if self.operator == "add":
            return text + visual
        if self.operator == "gated":
            assert self.gate is not None
            gate = torch.sigmoid(self.gate(torch.cat((text, visual), dim=-1)))
            return gate * text + (1.0 - gate) * visual
        return text.new_zeros((*text.shape[:-1], self.visual_size))

    def forward(self, hidden_states: Tensor, state: FusionState) -> dict[str, Tensor]:
        if self.operator == "disabled":
            return {
                target: hidden_states.new_zeros((*hidden_states.shape[:-1], layer.out_features))
                for target, layer in self.output.items()
            }
        updates: dict[str, Tensor] = {}
        for target, projection in self.output.items():
            joint = self._joint(hidden_states, state, target=target)
            updates[target] = self.scales[target].to(joint.dtype) * projection(joint)
        return updates

    def extra_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class VisualRMSNorm(nn.Module):
    """Visual-width RMS normalization with FP32 reduction and learned gain."""

    def __init__(
        self,
        width: int,
        eps: float,
        *,
        parameter_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=parameter_dtype))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        dtype = value.dtype
        normalized = value.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return self.weight.to(dtype) * normalized.to(dtype)
