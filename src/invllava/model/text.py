from __future__ import annotations

import torch
from torch import Tensor

from invllava.model.llama.generation import generate
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.sequence import build_text_only_fusion_state
from invllava.model.types import ExpandedSequence, LanguageModelOutput


def forward_text_only(
    model: InverseLlamaForCausalLM,
    input_ids: Tensor,
    *,
    feature_dim: int,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
) -> LanguageModelOutput:
    embeddings = model.model.embed_tokens(input_ids)
    state = build_text_only_fusion_state(embeddings, feature_dim, attention_mask)
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        fusion_state=state,
    )


def generate_text_only(
    model: InverseLlamaForCausalLM,
    input_ids: Tensor,
    *,
    feature_dim: int,
    attention_mask: Tensor | None = None,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> Tensor:
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    embeddings = model.model.embed_tokens(input_ids)
    sequence = ExpandedSequence(
        inputs_embeds=embeddings,
        attention_mask=attention_mask.bool(),
        position_ids=attention_mask.long().cumsum(-1).sub(1).clamp_min(0),
        labels=None,
        fusion_state=build_text_only_fusion_state(embeddings, feature_dim, attention_mask),
    )
    return generate(
        model,
        sequence,
        max_new_tokens=max_new_tokens,
        eos_token_id=model.config.eos_token_id,
        temperature=temperature,
        top_p=top_p,
    )
