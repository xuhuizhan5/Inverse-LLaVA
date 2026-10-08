from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class FusionState:
    """Visual branch aligned one-to-one with the expanded language sequence."""

    visual_features: Tensor
    text_mask: Tensor
    vision_mask: Tensor

    def validate(self, *, batch: int, sequence: int, feature_dim: int) -> None:
        if tuple(self.visual_features.shape) != (batch, sequence, feature_dim):
            raise ValueError(
                "visual feature shape mismatch: expected "
                f"{(batch, sequence, feature_dim)}, got {tuple(self.visual_features.shape)}"
            )
        for name, mask in (("text_mask", self.text_mask), ("vision_mask", self.vision_mask)):
            if tuple(mask.shape) != (batch, sequence):
                raise ValueError(f"{name} must have shape {(batch, sequence)}")
        overlap = (self.text_mask & self.vision_mask).any()
        if torch.compiler.is_compiling():
            # Preserve the invariant in the captured graph without a
            # host-synchronizing Tensor.item() graph break.
            torch._assert(~overlap, "text and vision masks must be disjoint")
        elif overlap.item():
            raise ValueError("text and vision masks must be disjoint")


@dataclass(frozen=True)
class ExpandedSequence:
    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    labels: Tensor | None
    fusion_state: FusionState


@dataclass(frozen=True)
class LayerKV:
    key: Tensor
    value: Tensor


@dataclass(frozen=True)
class LanguageModelOutput:
    logits: Tensor
    loss: Tensor | None
    past_key_values: tuple[LayerKV, ...] | None
    hidden_states: tuple[Tensor, ...] | None = None
    # Multimodal training exposure after expansion, truncation, and causal shift.
    # Counts are absent for inference and bare language-model calls.
    supervised_token_count: Tensor | None = None
    expanded_token_count: Tensor | None = None
