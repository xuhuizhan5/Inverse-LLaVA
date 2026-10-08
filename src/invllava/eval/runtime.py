from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from PIL import Image

from invllava.data.collate import (
    expand_to_square,
    image_mean_background,
    tokenize_with_image_placeholder,
)
from invllava.eval.records import PredictionRecord, PredictionStore
from invllava.eval.types import EvaluationExample, GenerationRequest, generation_request
from invllava.model.llama.generation import generate as generate_tokens
from invllava.model.modeling import (
    InverseLLaVAForConditionalGeneration,
    LLaVAReferenceForConditionalGeneration,
)
from invllava.model.types import ExpandedSequence


def _left_pad_token_rows(
    token_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize non-empty token rows with their final token right-aligned.

    Cached autoregressive decoding consumes the logits at the final physical
    position. Keeping the initial token batch left-padded makes that invariant
    explicit before multimodal expansion as well as after it.
    """

    if not token_rows:
        raise ValueError("cannot pad an empty token batch")
    if any(not row for row in token_rows):
        raise ValueError("token rows must be non-empty")
    width = max(map(len, token_rows))
    input_ids = torch.full(
        (len(token_rows), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, ids in enumerate(token_rows):
        start = width - len(ids)
        input_ids[index, start:] = torch.as_tensor(ids, dtype=torch.long, device=device)
        attention_mask[index, start:] = True
    return input_ids, attention_mask


class InverseGenerator:
    """Reference-free batched runtime with left-padded cached decoding."""

    def __init__(
        self,
        model: InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration,
        tokenizer: Any,
        image_processor: Any,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        pad_to_square: bool = True,
        cache_mode: str = "kv",
    ) -> None:
        if cache_mode not in {"kv", "full_recompute"}:
            raise ValueError("cache_mode must be kv or full_recompute")
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.pad_to_square = pad_to_square
        self.cache_mode = cache_mode
        self.model.padding_side = "left"

    @torch.inference_mode()
    def __call__(self, example: GenerationRequest) -> str:
        return self.generate_many((example,))[0]

    @torch.inference_mode()
    def generate_many(self, examples: Sequence[GenerationRequest]) -> list[str]:
        if not examples:
            return []
        expanded = self.prepare_many(examples)
        generated = generate_tokens(
            self.model.language_model,
            expanded,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            cache_mode=self.cache_mode,
        )
        return [self.tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]

    @torch.inference_mode()
    def prepare_many(self, examples: Sequence[GenerationRequest]) -> ExpandedSequence:
        """Preprocess and encode a batch without decoding (also used by analysis)."""

        if not examples:
            raise ValueError("cannot prepare an empty request batch")
        device = next(self.model.parameters()).device
        token_rows: list[list[int]] = []
        pixel_rows: list[list[torch.Tensor]] = []
        for example in examples:
            token_rows.append(
                tokenize_with_image_placeholder(
                    self.tokenizer, example.prompt, add_special_tokens=True
                )
            )
            pixels: list[torch.Tensor] = []
            for path in example.images:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if self.pad_to_square:
                        background = image_mean_background(self.image_processor.image_mean)
                        image = expand_to_square(image, background)
                    pixels.append(
                        self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
                    )
            if len(pixels) != example.prompt.count("<image>"):
                raise ValueError(f"image/prompt mismatch for evaluation sample {example.id}")
            if not pixels:
                raise ValueError("use the text-only runtime for examples without images")
            pixel_rows.append(pixels)
        input_ids, attention_mask = _left_pad_token_rows(
            token_rows,
            pad_token_id=self.tokenizer.pad_token_id,
            device=device,
        )
        counts = [len(row) for row in pixel_rows]
        flat_pixels = [pixel for row in pixel_rows for pixel in row]
        flat_features = self.model.encode_images(torch.stack(flat_pixels).to(device))
        feature_rows = []
        offset = 0
        for count in counts:
            feature_rows.append([flat_features[index] for index in range(offset, offset + count)])
            offset += count
        return self.model.prepare(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=feature_rows,
        )


def run_evaluation(
    examples: Iterable[EvaluationExample],
    *,
    generate: Callable[[GenerationRequest], str],
    store: PredictionStore,
    experiment_id: str,
    checkpoint_id: str,
    protocol_id: str,
    generation: dict[str, Any],
) -> int:
    written = 0
    for example in examples:
        if example.id in store.completed:
            continue
        prediction = generate(generation_request(example))
        store.append(
            PredictionRecord(
                schema_version=1,
                protocol_id=protocol_id,
                experiment_id=experiment_id,
                checkpoint_id=checkpoint_id,
                sample_id=example.id,
                prompt=example.prompt,
                prediction=prediction,
                references=example.references,
                image_ids=tuple(str(path) for path in example.images),
                generation=generation,
                metadata=example.metadata,
            )
        )
        written += 1
    return written


def run_batched_evaluation(
    examples: Sequence[EvaluationExample],
    *,
    generate: Callable[[Sequence[GenerationRequest]], list[str]],
    batch_size: int,
    store: PredictionStore,
    experiment_id: str,
    checkpoint_id: str,
    protocol_id: str,
    generation: dict[str, Any],
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    pending = [example for example in examples if example.id not in store.completed]
    written = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        requests = [generation_request(example) for example in chunk]
        predictions = generate(requests)
        if len(predictions) != len(chunk):
            raise RuntimeError("batched generator returned the wrong number of predictions")
        for example, prediction in zip(chunk, predictions, strict=True):
            store.append(
                PredictionRecord(
                    schema_version=1,
                    protocol_id=protocol_id,
                    experiment_id=experiment_id,
                    checkpoint_id=checkpoint_id,
                    sample_id=example.id,
                    prompt=example.prompt,
                    prediction=prediction,
                    references=example.references,
                    image_ids=tuple(str(path) for path in example.images),
                    generation=generation,
                    metadata=example.metadata,
                )
            )
            written += 1
    return written
