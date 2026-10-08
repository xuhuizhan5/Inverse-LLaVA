from pathlib import Path

import pytest
import torch

from invllava.data.controls import (
    collect_supervised_prefix,
    deterministic_image_derangement,
    prefix_ids,
    supervised_tokens_after_expansion,
    token_matched_prefix,
)
from invllava.data.types import ConversationSample, Turn
from invllava.model.sequence import expand_multimodal_sequence


@pytest.mark.parametrize("patches,max_length", [(1, 8), (4, 8), (576, 580), (576, 2048)])
def test_token_budget_matches_actual_expanded_causal_labels(patches: int, max_length: int) -> None:
    ids = [1, -200, 5, 6, 7, 8, 9, 10]
    labels = [-100, -100, -100, 6, 7, 8, 9, 10]
    expanded = expand_multimodal_sequence(
        input_ids=torch.tensor([ids]),
        token_embeddings=torch.zeros(1, len(ids), 2),
        image_features=[[torch.zeros(patches, 3)]],
        image_token_id=-200,
        labels=torch.tensor([labels]),
        max_length=max_length,
    )
    observed = int(expanded.labels[:, 1:].ne(-100).sum())
    assert (
        supervised_tokens_after_expansion(
            ids,
            labels,
            image_patch_counts=[patches],
            max_length=max_length,
        )
        == observed
    )


def test_token_budget_reports_all_truncated_targets_and_lost_placeholders() -> None:
    assert (
        supervised_tokens_after_expansion(
            [1, -200, 5],
            [-100, -100, 5],
            image_patch_counts=[576],
            max_length=16,
        )
        == 0
    )
    with pytest.raises(ValueError, match="placeholder"):
        supervised_tokens_after_expansion(
            [1, 2, -200, 5],
            [-100, -100, -100, 5],
            image_patch_counts=[576],
            max_length=2,
        )


def _sample(identifier: str, image: str) -> ConversationSample:
    return ConversationSample(
        identifier,
        (Path(image),),
        (Turn("user", "<image> question"), Turn("assistant", "answer")),
        "fixture",
    )


def test_token_matched_prefix_uses_closest_whole_sample_boundary() -> None:
    samples = [_sample("a", "a.jpg"), _sample("b", "b.jpg"), _sample("c", "c.jpg")]
    assert [
        sample.id
        for sample in token_matched_prefix(
            samples,
            {"a": 4, "b": 5, "c": 9},
            target_tokens=8,
        )
    ] == ["a", "b"]


@pytest.mark.parametrize("minimum_rows,target", [(2, 8), (1, 17), (3, 1)])
def test_lazy_supervised_prefix_matches_full_pool_selection(minimum_rows, target) -> None:
    samples = [_sample(str(i), f"{i}.jpg") for i in range(6)]
    counts = {str(i): count for i, count in enumerate((0, 4, 5, 9, 0, 100))}
    visited = []

    def count(sample):
        visited.append(sample.id)
        return counts[sample.id]

    prefix = collect_supervised_prefix(
        iter(samples), count, minimum_rows=minimum_rows, target_tokens=target
    )
    full = [sample for sample in samples if counts[sample.id] > 0]
    assert list(prefix.samples[:minimum_rows]) == full[:minimum_rows]
    assert token_matched_prefix(prefix.samples, prefix.token_counts, target_tokens=target) == (
        token_matched_prefix(full, counts, target_tokens=target)
    )
    assert len(visited) == prefix.examined_rows < len(samples)
    assert prefix.excluded_ids == ("0",)


def test_supervised_prefix_rejects_an_insufficient_pool() -> None:
    with pytest.raises(ValueError, match="cannot satisfy"):
        collect_supervised_prefix(
            [_sample("a", "a.jpg")], lambda _: 2, minimum_rows=2, target_tokens=1
        )


def test_derangement_is_deterministic_and_changes_every_binding() -> None:
    samples = [_sample("a", "a.jpg"), _sample("b", "b.jpg"), _sample("c", "c.jpg")]
    first = deterministic_image_derangement(samples, seed=17)
    second = deterministic_image_derangement(samples, seed=17)
    assert [sample.images for sample in first] == [sample.images for sample in second]
    assert all(samples[index].images != first[index].images for index in range(len(samples)))


def test_derangement_rejects_duplicate_content_binding() -> None:
    with pytest.raises(ValueError, match="do not admit"):
        deterministic_image_derangement(
            [_sample("a", "same.jpg"), _sample("b", "same.jpg")], seed=17
        )


def test_prefix_ids_preserves_content() -> None:
    sample = _sample("a", "a.jpg")
    prefixed = prefix_ids([sample], "paired")[0]
    assert prefixed.id == "paired:a"
    assert prefixed.images == sample.images
