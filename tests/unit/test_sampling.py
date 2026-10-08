from pathlib import Path

import torch

from invllava.data.sampling import nested_sample
from invllava.data.types import ConversationSample, Turn
from invllava.train.sampler import dataloader_generator


def samples(count: int) -> list[ConversationSample]:
    return [
        ConversationSample(str(index), (Path(f"{index}.png"),), (Turn("user", "<image>"),), "x")
        for index in range(count)
    ]


def test_nested_sample_is_nested_and_deterministic() -> None:
    values = samples(100)
    small = nested_sample(values, fraction=0.05, seed=17)
    large = nested_sample(values, fraction=0.2, seed=17)
    assert [item.id for item in small] == [
        item.id for item in nested_sample(values, fraction=0.05, seed=17)
    ]
    assert {item.id for item in small} < {item.id for item in large}


def test_dataloader_generator_is_repeatable_and_isolated() -> None:
    torch.manual_seed(91)
    expected_global = torch.rand(3)
    torch.manual_seed(91)
    first = dataloader_generator(42, rank=2)
    second = dataloader_generator(42, rank=2)
    torch.testing.assert_close(torch.rand(4, generator=first), torch.rand(4, generator=second))
    torch.testing.assert_close(torch.rand(3), expected_global)
