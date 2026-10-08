from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence, Sized

import torch
from torch.utils.data import Sampler


def dataloader_generator(seed: int, *, rank: int = 0) -> torch.Generator:
    """Keep DataLoader worker seeding isolated from the model RNG stream.

    The publication pipeline deliberately uses deterministic image transforms.
    A dedicated generator therefore makes iterator creation and worker startup
    reproducible without consuming the RNG restored for model training.
    """

    if rank < 0:
        raise ValueError("rank must be non-negative")
    generator = torch.Generator()
    generator.manual_seed(seed + rank)
    return generator


class EpochSeededSampler(Sampler[int]):
    def __init__(self, data_source: Sized, *, seed: int, rank: int = 0, replicas: int = 1) -> None:
        self.data_source = data_source
        self.seed = seed
        self.rank = rank
        self.replicas = replicas
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        indices = list(range(len(self.data_source)))
        random.Random(self.seed + self.epoch).shuffle(indices)
        padding = math.ceil(len(indices) / self.replicas) * self.replicas - len(indices)
        indices.extend(indices[:padding])
        return iter(indices[self.rank :: self.replicas])

    def __len__(self) -> int:
        return math.ceil(len(self.data_source) / self.replicas)


def _balanced_chunks(
    indices: list[int], lengths: Sequence[int], chunk_count: int
) -> list[list[int]]:
    """Balance long examples across the consecutive per-device batches."""

    if not indices:
        return []
    if len(indices) % chunk_count:
        return [indices[offset::chunk_count] for offset in range(chunk_count)]
    target_size = len(indices) // chunk_count
    chunks: list[list[int]] = [[] for _ in range(chunk_count)]
    totals = [0] * chunk_count
    for index in indices:
        available = [slot for slot, chunk in enumerate(chunks) if len(chunk) < target_size]
        slot = min(available, key=lambda value: totals[value])
        chunks[slot].append(index)
        totals[slot] += abs(lengths[index])
    return chunks


def modality_length_grouped_indices(
    lengths: Sequence[int],
    *,
    batch_size: int,
    group_count: int,
    seed: int,
    epoch: int,
) -> list[int]:
    """Return one deterministic, padding-aware order containing every index once.

    Positive lengths denote multimodal samples and negative lengths denote
    text-only samples. Full megabatches contain only one modality. Within a
    megabatch, long samples are balanced across ``group_count`` consecutive
    per-device batches, following the intent of LLaVA's training sampler.
    """

    if batch_size <= 0 or group_count <= 0:
        raise ValueError("batch_size and group_count must be positive")
    if any(length == 0 for length in lengths):
        raise ValueError("modality lengths must be non-zero")
    if not lengths:
        return []
    generator = torch.Generator().manual_seed(seed + epoch)
    megabatch_size = batch_size * group_count

    def modality_megabatches(pool: list[int]) -> tuple[list[list[int]], list[int]]:
        permutation = torch.randperm(len(pool), generator=generator).tolist()
        shuffled = [pool[index] for index in permutation]
        full: list[list[int]] = []
        remainder: list[int] = []
        for offset in range(0, len(shuffled), megabatch_size):
            group = shuffled[offset : offset + megabatch_size]
            ordered = sorted(group, key=lambda index: abs(lengths[index]), reverse=True)
            if len(group) == megabatch_size:
                chunks = _balanced_chunks(ordered, lengths, group_count)
                full.append([index for chunk in chunks for index in chunk])
            else:
                remainder.extend(ordered)
        return full, remainder

    visual = [index for index, length in enumerate(lengths) if length > 0]
    text = [index for index, length in enumerate(lengths) if length < 0]
    megabatches: list[list[int]] = []
    remainder: list[int] = []
    for pool in (visual, text):
        full, tail = modality_megabatches(pool)
        megabatches.extend(full)
        remainder.extend(tail)
    if megabatches:
        order = torch.randperm(len(megabatches), generator=generator).tolist()
        megabatches = [megabatches[index] for index in order]
    if remainder:
        remainder = sorted(remainder, key=lambda index: abs(lengths[index]), reverse=True)
        chunks = _balanced_chunks(remainder, lengths, min(group_count, len(remainder)))
        megabatches.append([index for chunk in chunks for index in chunk])
    result = [index for megabatch in megabatches for index in megabatch]
    if sorted(result) != list(range(len(lengths))):
        raise RuntimeError("grouped sampler did not preserve the dataset index set")
    return result


class ModalityLengthGroupedSampler(Sampler[int]):
    """Epoch-seeded sampler for efficient mixed visual/text instruction data."""

    def __init__(
        self,
        lengths: Sequence[int],
        *,
        batch_size: int,
        group_count: int,
        seed: int,
    ) -> None:
        self.lengths = tuple(lengths)
        self.batch_size = batch_size
        self.group_count = group_count
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        return iter(
            modality_length_grouped_indices(
                self.lengths,
                batch_size=self.batch_size,
                group_count=self.group_count,
                seed=self.seed,
                epoch=self.epoch,
            )
        )

    def __len__(self) -> int:
        return len(self.lengths)
