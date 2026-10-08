from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from invllava.model.sequence import build_text_only_fusion_state
from invllava.model.types import ExpandedSequence


@dataclass(frozen=True)
class TimingSummary:
    measurement: str
    repetitions: int
    warmups: int
    median_seconds: float
    q1_seconds: float
    q3_seconds: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AutoregressiveTimingSummary:
    measurement: str
    repetitions: int
    warmups: int
    batch_size: int
    prompt_tokens: int
    fixed_decode_tokens: int
    median_time_to_first_token_seconds: float
    q1_time_to_first_token_seconds: float
    q3_time_to_first_token_seconds: float
    median_decode_seconds: float
    q1_decode_seconds: float
    q3_decode_seconds: float
    median_decode_tokens_per_second: float
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def profile_callable(
    operation: Callable[[], object], *, warmups: int = 10, repetitions: int = 30
) -> TimingSummary:
    if repetitions < 3:
        raise ValueError("at least three repetitions are required")
    for _ in range(warmups):
        operation()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repetitions):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    quartiles = statistics.quantiles(samples, n=4, method="inclusive")
    return TimingSummary(
        measurement="measured",
        repetitions=repetitions,
        warmups=warmups,
        median_seconds=statistics.median(samples),
        q1_seconds=quartiles[0],
        q3_seconds=quartiles[2],
        peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
    )


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return statistics.median(values), quartiles[0], quartiles[2]


@torch.inference_mode()
def profile_autoregressive(
    language_model: Any,
    expanded: ExpandedSequence,
    *,
    fixed_decode_tokens: int = 32,
    fusion_feature_dim: int | None = None,
    warmups: int = 10,
    repetitions: int = 30,
    embed_tokens: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> AutoregressiveTimingSummary:
    """Measure prefill/first-token and fixed-length cached decode separately.

    EOS is deliberately ignored so architecture comparisons decode the same
    number of tokens. Image loading and vision encoding are outside this timing;
    profile them separately with ``profile_callable(generator.prepare_many)``.
    """

    if repetitions < 3 or warmups < 0 or fixed_decode_tokens <= 0:
        raise ValueError("profiling requires >=3 repetitions and positive decode length")
    batch = expanded.inputs_embeds.shape[0]

    def one() -> tuple[float, float]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function("invllava::prefill"):
            output = language_model(
                inputs_embeds=expanded.inputs_embeds,
                attention_mask=expanded.attention_mask,
                position_ids=expanded.position_ids,
                fusion_state=(expanded.fusion_state if fusion_feature_dim is not None else None),
                use_cache=True,
            )
        if output.past_key_values is None:
            raise RuntimeError("profiled language model returned no KV cache")
        last_indices = (
            torch.arange(
                expanded.attention_mask.shape[1],
                device=expanded.attention_mask.device,
            )
            .unsqueeze(0)
            .expand(batch, -1)
            .masked_fill(~expanded.attention_mask, -1)
            .max(dim=1)
            .values
        )
        rows = torch.arange(batch, device=last_indices.device)
        current = output.logits[rows, last_indices].argmax(dim=-1, keepdim=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        first_token = time.perf_counter()
        past = output.past_key_values
        attention = expanded.attention_mask
        with torch.profiler.record_function("invllava::fixed_decode"):
            for _ in range(fixed_decode_tokens):
                attention = torch.cat(
                    (
                        attention,
                        torch.ones((batch, 1), dtype=torch.bool, device=attention.device),
                    ),
                    dim=1,
                )
                position_ids = attention.long().sum(dim=-1, keepdim=True) - 1
                token_embeddings = (
                    embed_tokens(current)
                    if embed_tokens is not None
                    else language_model.model.embed_tokens(current)
                )
                fusion_state = (
                    build_text_only_fusion_state(token_embeddings, fusion_feature_dim)
                    if fusion_feature_dim is not None
                    else None
                )
                output = language_model(
                    inputs_embeds=token_embeddings,
                    attention_mask=attention,
                    position_ids=position_ids,
                    fusion_state=fusion_state,
                    past_key_values=past,
                    use_cache=True,
                )
                if output.past_key_values is None:
                    raise RuntimeError("decode step returned no KV cache")
                past = output.past_key_values
                current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        return first_token - start, end - first_token

    for _ in range(warmups):
        one()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    first_samples: list[float] = []
    decode_samples: list[float] = []
    for _ in range(repetitions):
        first, decode = one()
        first_samples.append(first)
        decode_samples.append(decode)
    first_median, first_q1, first_q3 = _quartiles(first_samples)
    decode_median, decode_q1, decode_q3 = _quartiles(decode_samples)
    return AutoregressiveTimingSummary(
        measurement="measured",
        repetitions=repetitions,
        warmups=warmups,
        batch_size=batch,
        prompt_tokens=int(expanded.attention_mask.sum().item()),
        fixed_decode_tokens=fixed_decode_tokens,
        median_time_to_first_token_seconds=first_median,
        q1_time_to_first_token_seconds=first_q1,
        q3_time_to_first_token_seconds=first_q3,
        median_decode_seconds=decode_median,
        q1_decode_seconds=decode_q1,
        q3_decode_seconds=decode_q3,
        median_decode_tokens_per_second=batch * fixed_decode_tokens / decode_median,
        peak_allocated_bytes=(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        peak_reserved_bytes=(
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        ),
    )
