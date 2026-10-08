from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaArchitecture:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf(cls, config: Any) -> LlamaArchitecture:
        return cls(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=getattr(config, "rope_theta", 10000.0),
            hidden_act=config.hidden_act,
            attention_dropout=getattr(config, "attention_dropout", 0.0),
            initializer_range=config.initializer_range,
            pad_token_id=config.pad_token_id or 0,
            bos_token_id=config.bos_token_id or 1,
            eos_token_id=config.eos_token_id or 2,
            tie_word_embeddings=getattr(config, "tie_word_embeddings", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
