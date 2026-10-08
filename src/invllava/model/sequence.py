"""Expand image placeholders into patch-aligned language and visual streams."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from invllava.model.types import ExpandedSequence, FusionState

IGNORE_INDEX = -100


def _as_image_lists(
    image_features: Sequence[Tensor | Sequence[Tensor]], batch: int
) -> list[list[Tensor]]:
    if len(image_features) != batch:
        raise ValueError(f"expected image features for {batch} samples, got {len(image_features)}")
    result: list[list[Tensor]] = []
    for value in image_features:
        result.append([value] if isinstance(value, Tensor) else list(value))
    return result


def expand_multimodal_sequence(
    *,
    input_ids: Tensor,
    token_embeddings: Tensor,
    image_features: Sequence[Tensor | Sequence[Tensor]],
    image_token_id: int,
    attention_mask: Tensor | None = None,
    labels: Tensor | None = None,
    max_length: int | None = None,
    visual_feature_dim: int | None = None,
    padding_side: str = "right",
    ignore_index: int = IGNORE_INDEX,
) -> ExpandedSequence:
    """Replace each image token with its patch sequence without mutable state.

    Text embeddings occupy text positions and are zero at patch positions. Visual
    features occupy patch positions and are zero at text positions. A sample may
    contain zero, one, or multiple images, but its placeholder/image counts must
    match exactly.
    """

    if input_ids.ndim != 2 or token_embeddings.ndim != 3:
        raise ValueError("input_ids must be [B,T] and token_embeddings [B,T,d_h]")
    batch, tokens = input_ids.shape
    if token_embeddings.shape[:2] != (batch, tokens):
        raise ValueError("token embeddings do not align with input_ids")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if labels is not None and labels.shape != input_ids.shape:
        raise ValueError("labels must have the same shape as input_ids")

    per_sample_images = _as_image_lists(image_features, batch)
    rows: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]] = []
    observed_dimensions = {
        image.shape[-1] for images in per_sample_images for image in images if image.ndim == 2
    }
    if len(observed_dimensions) > 1:
        raise ValueError("all images in a batch must share a visual feature dimension")
    observed = next(iter(observed_dimensions), None)
    if visual_feature_dim is not None and observed is not None and visual_feature_dim != observed:
        raise ValueError("configured and observed visual feature dimensions disagree")
    feature_dim: int | None = visual_feature_dim or observed

    for row in range(batch):
        keep = attention_mask[row]
        ids = input_ids[row][keep]
        embeds = token_embeddings[row][keep]
        row_labels = labels[row][keep] if labels is not None else None
        placeholders = int((ids == image_token_id).sum().item())
        images = per_sample_images[row]
        if placeholders != len(images):
            raise ValueError(
                f"sample {row} has {placeholders} image placeholders but "
                f"{len(images)} feature tensors"
            )

        text_parts: list[Tensor] = []
        visual_parts: list[Tensor] = []
        text_masks: list[Tensor] = []
        vision_masks: list[Tensor] = []
        label_parts: list[Tensor] = []
        image_index = 0

        for token_index, token_id in enumerate(ids.tolist()):
            if token_id != image_token_id:
                text_parts.append(embeds[token_index : token_index + 1])
                if feature_dim is None:
                    # Resolved when the first image is encountered. Text-only
                    # batches require callers to provide an empty [0,d_v] tensor.
                    for candidate in images:
                        feature_dim = candidate.shape[-1]
                        break
                visual_parts.append(embeds.new_zeros((1, feature_dim or 0), dtype=embeds.dtype))
                text_masks.append(torch.ones(1, dtype=torch.bool, device=embeds.device))
                vision_masks.append(torch.zeros(1, dtype=torch.bool, device=embeds.device))
                if row_labels is not None:
                    label_parts.append(row_labels[token_index : token_index + 1])
                continue

            visual = images[image_index]
            image_index += 1
            if visual.ndim != 2:
                raise ValueError("each image feature tensor must be [patches,d_v]")
            if feature_dim is None:
                feature_dim = visual.shape[-1]
                # Text parts created before the first placeholder have width 0.
                visual_parts = [
                    embeds.new_zeros((part.shape[0], feature_dim)) for part in text_parts
                ]
            elif visual.shape[-1] != feature_dim:
                raise ValueError("all images in a batch must share a visual feature dimension")
            patch_count = visual.shape[0]
            text_parts.append(embeds.new_zeros((patch_count, embeds.shape[-1])))
            visual_parts.append(visual.to(device=embeds.device, dtype=embeds.dtype))
            text_masks.append(torch.zeros(patch_count, dtype=torch.bool, device=embeds.device))
            vision_masks.append(torch.ones(patch_count, dtype=torch.bool, device=embeds.device))
            if row_labels is not None:
                label_parts.append(
                    torch.full(
                        (patch_count,),
                        ignore_index,
                        dtype=row_labels.dtype,
                        device=row_labels.device,
                    )
                )

        if feature_dim is None:
            raise ValueError(
                "feature dimension is ambiguous for a text-only batch; use "
                "build_text_only_fusion_state instead"
            )
        text = torch.cat(text_parts, dim=0)
        visual = torch.cat(
            [
                part
                if part.shape[-1] == feature_dim
                else part.new_zeros((part.shape[0], feature_dim))
                for part in visual_parts
            ],
            dim=0,
        )
        text_mask = torch.cat(text_masks)
        vision_mask = torch.cat(vision_masks)
        expanded_labels = torch.cat(label_parts) if row_labels is not None else None
        if max_length is not None:
            text = text[:max_length]
            visual = visual[:max_length]
            text_mask = text_mask[:max_length]
            vision_mask = vision_mask[:max_length]
            expanded_labels = expanded_labels[:max_length] if expanded_labels is not None else None
        rows.append((text, visual, text_mask, vision_mask, expanded_labels))

    assert feature_dim is not None
    length = max(row[0].shape[0] for row in rows)
    hidden_dim = token_embeddings.shape[-1]
    padded_text = token_embeddings.new_zeros((batch, length, hidden_dim))
    padded_visual = token_embeddings.new_zeros((batch, length, feature_dim))
    padded_attention = torch.zeros((batch, length), dtype=torch.bool, device=input_ids.device)
    padded_text_mask = torch.zeros_like(padded_attention)
    padded_vision_mask = torch.zeros_like(padded_attention)
    padded_labels = (
        torch.full((batch, length), ignore_index, dtype=labels.dtype, device=labels.device)
        if labels is not None
        else None
    )

    for index, (text, visual, text_mask, vision_mask, expanded_labels) in enumerate(rows):
        row_length = text.shape[0]
        offset = 0 if padding_side == "right" else length - row_length
        destination = slice(offset, offset + row_length)
        padded_text[index, destination] = text
        padded_visual[index, destination] = visual
        padded_attention[index, destination] = True
        padded_text_mask[index, destination] = text_mask
        padded_vision_mask[index, destination] = vision_mask
        if padded_labels is not None and expanded_labels is not None:
            padded_labels[index, destination] = expanded_labels

    # LLaVA's Vicuna-v1 compatibility path deliberately keeps a small number
    # of delimiter-ambiguous examples as context with every target masked.
    # Such a row is valid when another row in the optimizer microbatch carries
    # supervision. Reject only a fully unsupervised expanded batch, including
    # the case where multimodal expansion and truncation removed every target.
    if padded_labels is not None and not padded_labels.ne(ignore_index).any():
        raise ValueError("batch has no supervised token after multimodal sequence truncation")

    position_ids = padded_attention.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(~padded_attention, 0)
    fusion_state = FusionState(padded_visual, padded_text_mask, padded_vision_mask)
    fusion_state.validate(batch=batch, sequence=length, feature_dim=feature_dim)
    return ExpandedSequence(
        inputs_embeds=padded_text,
        attention_mask=padded_attention,
        position_ids=position_ids,
        labels=padded_labels,
        fusion_state=fusion_state,
    )


def build_text_only_fusion_state(
    hidden_states: Tensor,
    feature_dim: int,
    attention_mask: Tensor | None = None,
) -> FusionState:
    batch, sequence, _ = hidden_states.shape
    if attention_mask is None:
        text_mask = torch.ones((batch, sequence), dtype=torch.bool, device=hidden_states.device)
    else:
        if tuple(attention_mask.shape) != (batch, sequence):
            raise ValueError("text-only attention mask does not align with hidden states")
        text_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    return FusionState(
        visual_features=hidden_states.new_zeros((batch, sequence, feature_dim)),
        text_mask=text_mask,
        vision_mask=torch.zeros((batch, sequence), dtype=torch.bool, device=hidden_states.device),
    )
