from __future__ import annotations

import torch
from torch import Tensor

from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.sequence import build_text_only_fusion_state
from invllava.model.types import ExpandedSequence, FusionState


def _next_token(logits: Tensor, *, temperature: float, top_p: float) -> Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = probabilities.sort(descending=True)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities.masked_fill_(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1, sorted_indices, sorted_probabilities
        )
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    return torch.multinomial(probabilities, 1).squeeze(1)


def _generate_full_recompute(
    model: InverseLlamaForCausalLM,
    initial: ExpandedSequence,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float,
    top_p: float,
) -> Tensor:
    """Recompute the complete sequence at each step for paper-runtime parity."""

    embeddings = initial.inputs_embeds
    attention_mask = initial.attention_mask
    position_ids = initial.position_ids
    visual_features = initial.fusion_state.visual_features
    text_mask = initial.fusion_state.text_mask
    vision_mask = initial.fusion_state.vision_mask
    generated: list[Tensor] = []
    finished = torch.zeros(embeddings.shape[0], dtype=torch.bool, device=embeddings.device)

    for _ in range(max_new_tokens):
        output = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            fusion_state=FusionState(
                visual_features=visual_features,
                text_mask=text_mask,
                vision_mask=vision_mask,
            ),
            use_cache=False,
        )
        token = _next_token(output.logits[:, -1], temperature=temperature, top_p=top_p)
        token = torch.where(finished, torch.full_like(token, eos_token_id), token)
        generated.append(token)
        finished |= token.eq(eos_token_id)
        if finished.all():
            break

        current_embeddings = model.model.embed_tokens(token[:, None])
        embeddings = torch.cat((embeddings, current_embeddings), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=attention_mask.device,
                ),
            ),
            dim=1,
        )
        current_positions = attention_mask.long().sum(dim=-1, keepdim=True) - 1
        position_ids = torch.cat((position_ids, current_positions), dim=1)
        current_state = build_text_only_fusion_state(current_embeddings, visual_features.shape[-1])
        visual_features = torch.cat((visual_features, current_state.visual_features), dim=1)
        text_mask = torch.cat((text_mask, current_state.text_mask), dim=1)
        vision_mask = torch.cat((vision_mask, current_state.vision_mask), dim=1)

    return (
        torch.stack(generated, dim=1)
        if generated
        else embeddings.new_empty((embeddings.shape[0], 0), dtype=torch.long)
    )


@torch.inference_mode()
def generate(
    model: InverseLlamaForCausalLM,
    initial: ExpandedSequence,
    *,
    max_new_tokens: int,
    eos_token_id: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    cache_mode: str = "kv",
) -> Tensor:
    """Small deterministic-first generation loop with an explicit cache policy."""

    if cache_mode not in {"kv", "full_recompute"}:
        raise ValueError("cache_mode must be kv or full_recompute")
    if cache_mode == "full_recompute":
        return _generate_full_recompute(
            model,
            initial,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_p=top_p,
        )

    output = model(
        inputs_embeds=initial.inputs_embeds,
        attention_mask=initial.attention_mask,
        position_ids=initial.position_ids,
        fusion_state=initial.fusion_state,
        use_cache=True,
    )
    past = output.past_key_values
    attention_mask = initial.attention_mask
    generated: list[Tensor] = []
    finished = torch.zeros(output.logits.shape[0], dtype=torch.bool, device=output.logits.device)
    next_logits = output.logits[:, -1]
    feature_dim = initial.fusion_state.visual_features.shape[-1]

    for _ in range(max_new_tokens):
        next_token = _next_token(next_logits, temperature=temperature, top_p=top_p)
        next_token = torch.where(
            finished,
            torch.full_like(next_token, eos_token_id),
            next_token,
        )
        generated.append(next_token)
        finished |= next_token.eq(eos_token_id)
        if finished.all():
            break
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1), dtype=torch.bool, device=attention_mask.device
                ),
            ),
            dim=1,
        )
        current_embeddings = model.model.embed_tokens(next_token[:, None])
        current_state = build_text_only_fusion_state(current_embeddings, feature_dim)
        current_positions = attention_mask.long().sum(dim=-1, keepdim=True) - 1
        output = model(
            input_ids=next_token[:, None],
            attention_mask=attention_mask,
            position_ids=current_positions,
            fusion_state=current_state,
            past_key_values=past,
            use_cache=True,
        )
        past = output.past_key_values
        next_logits = output.logits[:, -1]
    return (
        torch.stack(generated, dim=1)
        if generated
        else output.logits.new_empty((0, 0), dtype=torch.long)
    )
