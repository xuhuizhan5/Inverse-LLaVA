from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch import Tensor

from invllava.data.types import ConversationSample
from invllava.prompting import format_vicuna_v1

IMAGE_TOKEN_ID = -200
IGNORE_INDEX = -100


def expand_to_square(image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    result = Image.new(image.mode, (size, size), background)
    result.paste(image, ((size - width) // 2, (size - height) // 2))
    return result


def image_mean_background(image_mean: Sequence[float]) -> tuple[int, int, int]:
    """Convert processor means to the integer RGB padding used by LLaVA."""

    if len(image_mean) != 3:
        raise ValueError("image processor mean must contain exactly three channels")
    return tuple(int(255 * value) for value in image_mean)  # type: ignore[return-value]


def tokenize_with_image_placeholder(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> list[int]:
    """Apply LLaVA's sentinel tokenizer without entering -200 in the vocabulary."""

    chunks = [
        # Encoding is intentionally untruncated here so round boundaries can
        # be validated before the collator applies its explicit 2,048-token
        # limit. Transformers otherwise emits a misleading model-input warning
        # for the bounded longest-prompt preflight.
        tokenizer(chunk, add_special_tokens=add_special_tokens, verbose=False).input_ids
        for chunk in text.split("<image>")
    ]
    offset = int(
        add_special_tokens
        and bool(chunks)
        and bool(chunks[0])
        and chunks[0][0] == tokenizer.bos_token_id
    )
    ids: list[int] = []
    if offset:
        ids.append(chunks[0][0])
    for index, chunk_ids in enumerate(chunks):
        ids.extend(chunk_ids[offset:])
        if index < len(chunks) - 1:
            ids.append(IMAGE_TOKEN_ID)
    return ids


def encode_vicuna_v1(tokenizer: Any, turns: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Reproduce LLaVA-1.5 ``preprocess_v1`` tokenization and target masking."""

    prompt = format_vicuna_v1(turns)
    input_ids = tokenize_with_image_placeholder(tokenizer, prompt)
    if not input_ids or input_ids[0] != tokenizer.bos_token_id:
        raise ValueError("Vicuna-v1 compatibility requires one leading BOS token")
    separator = " ASSISTANT: "
    round_lengths: list[tuple[int, int]] = []
    for conversation_round in prompt.split("</s>"):
        if not conversation_round:
            break
        parts = conversation_round.split(separator)
        if len(parts) != 2:
            # Match LLaVA-1.5 preprocess_v1: an ambiguous serialized round is
            # retained as input context while every target token is ignored.
            # The dataset index records these samples before model allocation.
            return input_ids, [IGNORE_INDEX] * len(input_ids)
        instruction = parts[0] + separator
        round_length = len(tokenize_with_image_placeholder(tokenizer, conversation_round))
        instruction_length = len(tokenize_with_image_placeholder(tokenizer, instruction)) - 2
        round_lengths.append((round_length, instruction_length))

    # LLaVA's original compatibility branch subtracts one token from every
    # round after the first for a subset of SentencePiece/tokenizers releases.
    # ``legacy`` and package-version metadata no longer predict that behavior
    # reliably. Select between the two historical layouts using the invariant
    # both branches were intended to preserve: the rounds must account for the
    # exact full-prompt tokenization.
    raw_cursor = 1 + sum(round_length for round_length, _ in round_lengths)
    adjusted_cursor = raw_cursor - max(len(round_lengths) - 1, 0)
    if raw_cursor == len(input_ids):
        adjust_later_rounds = False
    elif adjusted_cursor == len(input_ids):
        adjust_later_rounds = True
    else:
        raise ValueError(
            "Vicuna-v1 tokenization mismatch: "
            f"unadjusted rounds account for {raw_cursor} tokens and adjusted rounds "
            f"account for {adjusted_cursor}, while the full prompt has {len(input_ids)}"
        )

    labels = input_ids.copy()
    labels[0] = IGNORE_INDEX
    cursor = 1
    for round_index, (round_length, instruction_length) in enumerate(round_lengths):
        if adjust_later_rounds and round_index != 0:
            round_length -= 1
            instruction_length -= 1
        if instruction_length < 0 or round_length <= 0:
            raise ValueError("Vicuna-v1 tokenization produced invalid round boundaries")
        labels[cursor : cursor + instruction_length] = [IGNORE_INDEX] * instruction_length
        cursor += round_length
    labels[cursor:] = [IGNORE_INDEX] * (len(labels) - cursor)
    return input_ids, labels


@dataclass
class MultimodalCollator:
    tokenizer: Any
    image_transform: Callable[[Image.Image], Tensor]
    max_length: int = 2048

    def __call__(self, samples: Sequence[ConversationSample]) -> dict[str, object]:
        token_rows: list[Tensor] = []
        label_rows: list[Tensor] = []
        image_rows: list[list[Tensor]] = []
        for sample in samples:
            sample.validate()
            ids, supervised = encode_vicuna_v1(self.tokenizer, sample.turns)
            ids = ids[: self.max_length]
            supervised = supervised[: self.max_length]
            retained_images = sum(token == IMAGE_TOKEN_ID for token in ids)
            if retained_images != len(sample.images):
                raise ValueError(
                    f"sample {sample.id} truncation retained {retained_images} image tokens "
                    f"for {len(sample.images)} images"
                )
            token_rows.append(torch.tensor(ids, dtype=torch.long))
            label_rows.append(torch.tensor(supervised, dtype=torch.long))
            images: list[Tensor] = []
            for path in sample.images:
                with Image.open(path) as image:
                    images.append(self.image_transform(image.convert("RGB")))
            image_rows.append(images)
        width = max(row.numel() for row in token_rows)
        input_ids = torch.full((len(samples), width), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(samples), width), IGNORE_INDEX, dtype=torch.long)
        attention = torch.zeros((len(samples), width), dtype=torch.bool)
        for index, (ids, row_labels) in enumerate(zip(token_rows, label_rows, strict=True)):
            input_ids[index, : ids.numel()] = ids
            labels[index, : ids.numel()] = row_labels
            attention[index, : ids.numel()] = True
        if not bool(labels.ne(IGNORE_INDEX).any()):
            raise ValueError(
                "batch has no supervised assistant token; "
                f"sample_ids={[sample.id for sample in samples]}"
            )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
            "pixel_values": image_rows,
            "sample_ids": [sample.id for sample in samples],
        }
