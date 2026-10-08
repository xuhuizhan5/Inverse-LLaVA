from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.decoder import InverseLlamaDecoderLayer, RMSNorm
from invllava.model.types import FusionState, LanguageModelOutput, LayerKV


class InverseLlamaModel(nn.Module):
    def __init__(
        self,
        config: LlamaArchitecture,
        *,
        fusion_layers: Mapping[int, FusionBlock] | None = None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        fusion_layers = fusion_layers or {}
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [
                InverseLlamaDecoderLayer(
                    config, fusion=fusion_layers.get(index), backend=attention_backend
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        fusion_state: FusionState | None = None,
        past_key_values: tuple[LayerKV, ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, tuple[LayerKV, ...] | None, tuple[Tensor, ...] | None]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, sequence, _ = hidden_states.shape
        past_length = past_key_values[0].key.shape[2] if past_key_values else 0
        if position_ids is None:
            position_ids = (
                torch.arange(past_length, past_length + sequence, device=hidden_states.device)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        if attention_mask is not None:
            expected_mask = (batch, past_length + sequence)
            if attention_mask.shape != expected_mask:
                raise ValueError(
                    f"attention mask must be {expected_mask}, got {tuple(attention_mask.shape)}"
                )
            # Preserve None as the no-padding signal so every decoder layer can
            # use SDPA's fused causal path instead of materializing a mask.
            if not torch.compiler.is_compiling() and bool(attention_mask.all()):
                attention_mask = None
        if fusion_state is not None:
            # State describes the current query segment, not cached keys.
            fusion_state.validate(
                batch=batch,
                sequence=sequence,
                feature_dim=fusion_state.visual_features.shape[-1],
            )

        hidden_history: list[Tensor] | None = [] if output_hidden_states else None
        presents: list[LayerKV] | None = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            if hidden_history is not None:
                hidden_history.append(hidden_states)
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError(
                        "gradient checkpointing and KV caching cannot be enabled together"
                    )

                def checkpointed(value: Tensor, current_layer: nn.Module = layer) -> Tensor:
                    return current_layer(
                        value,
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        fusion_state=fusion_state,
                        past_key_value=None,
                        use_cache=False,
                    )[0]

                hidden_states = checkpoint(checkpointed, hidden_states, use_reentrant=False)
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    fusion_state=fusion_state,
                    past_key_value=past_key_values[index] if past_key_values else None,
                    use_cache=use_cache,
                )
            if presents is not None:
                assert present is not None
                presents.append(present)
        hidden_states = self.norm(hidden_states)
        if hidden_history is not None:
            hidden_history.append(hidden_states)
        return (
            hidden_states,
            tuple(presents) if presents is not None else None,
            tuple(hidden_history) if hidden_history is not None else None,
        )


class InverseLlamaForCausalLM(nn.Module):
    def __init__(
        self,
        config: LlamaArchitecture,
        *,
        fusion_layers: Mapping[int, FusionBlock] | None = None,
        attention_backend: str = "sdpa",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = InverseLlamaModel(
            config, fusion_layers=fusion_layers, attention_backend=attention_backend
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        fusion_state: FusionState | None = None,
        past_key_values: tuple[LayerKV, ...] | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
    ) -> LanguageModelOutput:
        hidden, present, hidden_history = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            fusion_state=fusion_state,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                raise ValueError("labels do not align with language-model sequence")
            loss = functional.cross_entropy(
                logits[:, :-1].contiguous().float().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return LanguageModelOutput(logits, loss, present, hidden_history)
