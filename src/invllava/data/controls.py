"""Deterministic continuation controls built from sealed prepared datasets."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from invllava.data.types import ConversationSample


@dataclass(frozen=True)
class SupervisedPrefix:
    samples: tuple[ConversationSample, ...]
    token_counts: dict[str, int]
    excluded_ids: tuple[str, ...]
    examined_rows: int


def collect_supervised_prefix(
    ordered_samples: Iterable[ConversationSample],
    count_tokens: Callable[[ConversationSample], int],
    *,
    minimum_rows: int,
    target_tokens: int,
) -> SupervisedPrefix:
    """Read only the deterministic prefix needed for row and token controls.

    With positive counts, a closest-token prefix cannot improve after the
    cumulative count reaches its target. Reading later rows cannot change
    either this match or the requested equal-row prefix.
    """

    if minimum_rows <= 0 or target_tokens <= 0:
        raise ValueError("row and token budgets must be positive")
    selected, excluded = [], []
    counts: dict[str, int] = {}
    seen: set[str] = set()
    total = examined = 0
    for sample in ordered_samples:
        if sample.id in seen:
            raise ValueError(f"duplicate control sample identity: {sample.id}")
        seen.add(sample.id)
        examined += 1
        count = count_tokens(sample)
        if count < 0:
            raise ValueError("supervised-token count cannot be negative")
        if count == 0:
            excluded.append(sample.id)
            continue
        selected.append(sample)
        counts[sample.id] = count
        total += count
        if len(selected) >= minimum_rows and total >= target_tokens:
            return SupervisedPrefix(tuple(selected), counts, tuple(excluded), examined)
    raise ValueError("instruction pool cannot satisfy the row and token control budgets")


def supervised_tokens_after_expansion(
    input_ids: Sequence[int],
    labels: Sequence[int],
    *,
    image_patch_counts: Sequence[int],
    max_length: int,
    image_token_id: int = -200,
    ignore_index: int = -100,
) -> int:
    """Count causal targets surviving collation and image-slot expansion.

    This metadata-only calculation follows the two truncations in the real
    collator/sequence path. Differential tests bind it to tensor expansion.
    """

    if len(input_ids) != len(labels) or max_length <= 0:
        raise ValueError("token/label lengths must agree and max_length must be positive")
    if any(count <= 0 for count in image_patch_counts):
        raise ValueError("each image must have a positive patch count")
    ids = input_ids[:max_length]
    if sum(token == image_token_id for token in ids) != len(image_patch_counts):
        raise ValueError("image placeholder count disagrees after collation truncation")
    patches = iter(image_patch_counts)
    position = 0
    supervised = 0
    for token, label in zip(ids, labels[:max_length], strict=True):
        if token == image_token_id:
            position += next(patches)
        else:
            supervised += int(0 < position < max_length and label != ignore_index)
            position += 1
    return supervised


def token_matched_prefix(
    samples: Sequence[ConversationSample],
    token_counts: Mapping[str, int],
    *,
    target_tokens: int,
) -> list[ConversationSample]:
    """Select the deterministic prefix closest to a target token exposure."""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    selected: list[ConversationSample] = []
    total = 0
    for sample in samples:
        count = token_counts.get(sample.id)
        if count is None or count <= 0:
            raise ValueError(f"sample {sample.id} lacks a positive supervised-token count")
        previous_error = abs(target_tokens - total)
        next_error = abs(target_tokens - (total + count))
        if total >= target_tokens or (selected and next_error > previous_error):
            break
        selected.append(sample)
        total += count
    if not selected:
        raise ValueError("token matching selected no samples")
    return selected


def deterministic_image_derangement(
    samples: Sequence[ConversationSample], *, seed: int
) -> list[ConversationSample]:
    """Rotate a seeded image ordering while forbidding any unchanged path binding."""

    if len(samples) < 2:
        raise ValueError("image derangement requires at least two samples")
    if any(len(sample.images) != 1 for sample in samples):
        raise ValueError("image derangement requires exactly one image per sample")
    ordered = sorted(
        range(len(samples)),
        key=lambda index: hashlib.sha256(f"{seed}:{samples[index].id}".encode()).hexdigest(),
    )
    source_images = [samples[index].images[0].resolve() for index in ordered]
    for offset in range(1, len(ordered)):
        rotated = source_images[offset:] + source_images[:offset]
        if all(
            samples[index].images[0].resolve() != rotated[position]
            for position, index in enumerate(ordered)
        ):
            replacements = {index: rotated[position] for position, index in enumerate(ordered)}
            return [
                replace(sample, images=(replacements[index],))
                for index, sample in enumerate(samples)
            ]
    raise ValueError("selected rows do not admit a path-level image derangement")


def prefix_ids(samples: Sequence[ConversationSample], prefix: str) -> list[ConversationSample]:
    if not prefix or ":" in prefix:
        raise ValueError("prefix must be non-empty and must not contain a colon")
    return [replace(sample, id=f"{prefix}:{sample.id}") for sample in samples]
