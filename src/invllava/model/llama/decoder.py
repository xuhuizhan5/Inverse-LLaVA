from __future__ import annotations

import torch
from torch import Tensor, nn

from invllava.model.fusion import FusionBlock
from invllava.model.llama.attention import InverseLlamaAttention
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.types import FusionState, LayerKV


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        dtype = hidden_states.dtype
        value = hidden_states.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * value.to(dtype)


class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaArchitecture) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(value)) * self.up_proj(value))


class InverseLlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: LlamaArchitecture,
        *,
        fusion: FusionBlock | None = None,
        backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.self_attn = InverseLlamaAttention(config, fusion=fusion, backend=backend)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        fusion_state: FusionState | None,
        past_key_value: LayerKV | None,
        use_cache: bool,
    ) -> tuple[Tensor, LayerKV | None]:
        residual = hidden_states
        attention, present = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids=position_ids,
            attention_mask=attention_mask,
            fusion_state=fusion_state,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attention
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states)), present
