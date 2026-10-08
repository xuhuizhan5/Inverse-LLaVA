from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.types import FusionState, LayerKV


def rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Unscaled RoPE with an explicit, checkpoint-compatible precision policy."""

    def __init__(
        self, head_dim: int, theta: float, *, precision: Literal["model", "float32"] = "model"
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 or not math.isfinite(theta) or theta <= 0:
            raise ValueError("RoPE needs a positive even head width and positive finite theta")
        self.head_dim = head_dim
        self.theta = theta
        self.precision = "model"
        self.register_buffer("inv_freq", self._float32_frequencies())
        self.set_precision(precision)

    def _float32_frequencies(self) -> Tensor:
        # Compute on CPU once at construction/loading, independent of the model
        # factory dtype and GPU matmul precision. Never round and then upcast.
        indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32, device="cpu")
        return 1.0 / (self.theta ** (indices / self.head_dim))

    def set_precision(self, precision: Literal["model", "float32"]) -> None:
        if precision not in {"model", "float32"}:
            raise ValueError("rotary precision must be model or float32")
        if precision == "float32":
            expected = self._float32_frequencies().to(self.inv_freq.device)
            if not torch.equal(self.inv_freq, expected.to(self.inv_freq.dtype)):
                raise ValueError("checkpoint rotary frequencies do not match unscaled RoPE")
            self.inv_freq = expected
        self.precision = precision

    def _apply(self, fn: Any, recurse: bool = True) -> RotaryEmbedding:
        original = self.inv_freq
        super()._apply(fn, recurse=recurse)
        if self.precision == "float32":
            # Module.to()/half()/bfloat16(), including distributed wrappers,
            # must move the original values without rounding this buffer.
            self.inv_freq = original.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def _load_from_state_dict(
        self,
        state_dict: Any,
        prefix: str,
        local_metadata: Any,
        strict: bool,
        missing_keys: Any,
        unexpected_keys: Any,
        error_msgs: Any,
    ) -> None:
        frequency = state_dict.get(prefix + "inv_freq")
        if self.precision == "float32" and frequency is not None:
            expected = self._float32_frequencies().to(frequency.device)
            if frequency.dtype != torch.float32 or not torch.equal(frequency, expected):
                error_msgs.append(
                    f"{prefix}inv_freq: FP32 RoPE cannot resume rounded or altered frequencies"
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if self.precision == "float32" and self.inv_freq.dtype != torch.float32:
            raise RuntimeError("FP32 rotary buffer was recast after loading")
        frequencies = torch.einsum("bi,j->bij", positions.float(), self.inv_freq)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype), embedding.sin().to(dtype)


def apply_rotary(query: Tensor, key: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_key_value(value: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return value
    batch, heads, sequence, dimension = value.shape
    value = value[:, :, None, :, :].expand(batch, heads, repeats, sequence, dimension)
    return value.reshape(batch, heads * repeats, sequence, dimension)


class InverseLlamaAttention(nn.Module):
    def __init__(
        self,
        config: LlamaArchitecture,
        *,
        fusion: FusionBlock | None = None,
        backend: str = "sdpa",
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size must divide attention heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("attention heads must divide key/value heads")
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.backend = backend
        if backend not in {"sdpa", "eager"}:
            raise ValueError("attention backend must be sdpa or eager")
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)
        self.fusion = fusion

    def forward(
        self,
        hidden_states: Tensor,
        *,
        position_ids: Tensor,
        attention_mask: Tensor | None,
        fusion_state: FusionState | None,
        past_key_value: LayerKV | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKV | None]:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if self.fusion is not None:
            if fusion_state is None:
                raise ValueError("a configured fusion layer requires FusionState")
            updates = self.fusion(hidden_states, fusion_state)
            query = query + updates.get("q", 0.0)
            key = key + updates.get("k", 0.0)
            value = value + updates.get("v", 0.0)

        query = query.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(position_ids, query.dtype)
        query, key = apply_rotary(query, key, cos, sin)
        if past_key_value is not None:
            key = torch.cat((past_key_value.key, key), dim=2)
            value = torch.cat((past_key_value.value, value), dim=2)
        present = LayerKV(key=key, value=value) if use_cache else None

        expanded_key = repeat_key_value(key, self.kv_groups)
        expanded_value = repeat_key_value(value, self.kv_groups)
        key_length = expanded_key.shape[2]
        past_length = key_length - query_length
        if attention_mask is not None and attention_mask.shape != (batch, key_length):
            raise ValueError(
                f"attention mask must be {(batch, key_length)}, got {tuple(attention_mask.shape)}"
            )

        # SDPA can use its fused causal kernel when the initial sequence has no
        # padding. Padding and cached multi-token segments need an explicit mask,
        # but a boolean mask is sufficient; avoid a second sequence-squared
        # floating-point allocation.
        fused_causal = self.backend == "sdpa" and attention_mask is None and past_length == 0
        allowed: Tensor | None = None
        if not fused_causal and not (attention_mask is None and query_length == 1):
            query_positions = torch.arange(query_length, device=query.device) + past_length
            key_positions = torch.arange(key_length, device=query.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            allowed = allowed[None, None, :, :]
            if attention_mask is not None:
                allowed = allowed & attention_mask[:, None, None, :].bool()

        if self.backend == "eager":
            weights = torch.matmul(query, expanded_key.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if allowed is not None:
                weights.masked_fill_(~allowed, torch.finfo(weights.dtype).min)
            weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            weights = functional.dropout(weights, p=self.dropout, training=self.training)
            output = torch.matmul(weights, expanded_value)
        else:
            output = functional.scaled_dot_product_attention(
                query,
                expanded_key,
                expanded_value,
                attn_mask=allowed,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=fused_causal,
            )
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.o_proj(output), present
