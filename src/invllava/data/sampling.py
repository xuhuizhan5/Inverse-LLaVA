from __future__ import annotations

import hashlib
from collections.abc import Iterable

from invllava.data.types import ConversationSample


def _rank(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()


def nested_sample(
    samples: Iterable[ConversationSample], *, fraction: float, seed: int, maximum: int | None = None
) -> list[ConversationSample]:
    """Stable hash sampling makes every smaller fraction a subset of a larger one."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0,1]")
    ordered = sorted(samples, key=lambda sample: _rank(sample.id, seed))
    count = max(1, round(len(ordered) * fraction)) if ordered else 0
    if maximum is not None:
        count = min(count, maximum)
    return ordered[:count]


def stratified_nested_indices(
    ids: list[str],
    strata: list[str],
    *,
    fraction: float,
    seed: int,
    maximum: int | None = None,
) -> list[int]:
    if len(ids) != len(strata) or not 0 < fraction <= 1:
        raise ValueError("IDs/strata must align and fraction must be in (0,1]")
    groups: dict[str, list[int]] = {}
    for index, stratum in enumerate(strata):
        groups.setdefault(stratum, []).append(index)
    selected: list[int] = []
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: _rank(ids[index], seed))
        count = max(1, round(len(ordered) * fraction)) if ordered else 0
        selected.extend(ordered[:count])
    selected.sort(key=lambda index: _rank(ids[index], seed))
    if maximum is not None:
        selected = selected[:maximum]
    return selected
