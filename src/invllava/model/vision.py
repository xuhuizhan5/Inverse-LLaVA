from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn


def _validate_clip_loading_info(loading_info: dict[str, Any]) -> None:
    allowed_unexpected = (
        "text_model.",
        "text_projection.",
        "visual_projection.",
    )
    unexpected = tuple(
        key
        for key in loading_info.get("unexpected_keys", ())
        if key != "logit_scale" and not key.startswith(allowed_unexpected)
    )
    missing = tuple(loading_info.get("missing_keys", ()))
    mismatched = tuple(loading_info.get("mismatched_keys", ()))
    errors = tuple(loading_info.get("error_msgs", ()))
    if missing or unexpected or mismatched or errors:
        raise RuntimeError(
            "CLIP vision checkpoint load is incomplete: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}, errors={errors}"
        )


class VisionEncoder(nn.Module):
    """Thin CLIP wrapper with explicit feature-layer and token selection."""

    def __init__(
        self,
        tower: nn.Module,
        *,
        feature_layers: Sequence[int] = (-1,),
        feature_select: str = "patch",
        freeze: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not feature_layers:
            raise ValueError("at least one vision feature layer is required")
        if feature_select not in {"patch", "cls_patch"}:
            raise ValueError("feature_select must be patch or cls_patch")
        self.tower = tower
        if dtype is not None:
            self.tower.to(dtype=dtype)
        self.feature_layers = tuple(feature_layers)
        self.feature_select = feature_select
        self.frozen = freeze
        if freeze:
            self.tower.requires_grad_(False)
            self.tower.eval()

    def train(self, mode: bool = True) -> VisionEncoder:
        super().train(mode)
        if self.frozen:
            self.tower.eval()
        return self

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        revision: str | None = None,
        attention_backend: str = "sdpa",
        local_files_only: bool = False,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> VisionEncoder:
        from transformers import CLIPVisionModel
        from transformers.utils import logging as transformers_logging

        if attention_backend not in {"sdpa", "eager"}:
            raise ValueError("vision attention backend must be sdpa or eager")
        verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            tower, loading_info = CLIPVisionModel.from_pretrained(
                checkpoint,
                revision=revision,
                attn_implementation=attention_backend,
                local_files_only=local_files_only,
                token=token,
                output_loading_info=True,
            )
        finally:
            transformers_logging.set_verbosity(verbosity)
        _validate_clip_loading_info(loading_info)
        return cls(tower, **kwargs)

    def forward(self, pixel_values: Tensor) -> Tensor:
        tower_parameter = next(self.tower.parameters())
        pixel_values = pixel_values.to(device=tower_parameter.device, dtype=tower_parameter.dtype)
        context = (
            torch.no_grad()
            if not any(p.requires_grad for p in self.tower.parameters())
            else nullcontext()
        )
        with context:
            output = self.tower(pixel_values=pixel_values, output_hidden_states=True)
        selected = [output.hidden_states[index] for index in self.feature_layers]
        if self.feature_select == "patch":
            selected = [features[:, 1:] for features in selected]
        return torch.cat(selected, dim=-1)
