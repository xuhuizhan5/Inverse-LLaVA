import pytest

from scripts.audit_training_exposure import distributed_epoch_exposure


def test_distributed_exposure_counts_padding_and_causal_targets():
    targets = [2, 3, 5, 7, 11]
    result = distributed_epoch_exposure(
        [1, 2, 3, 4, 5], targets, batch_size=2, processes=2, accumulation=2, seed=42
    )
    assert result["unique_rows"] == 5
    assert result["sampled_rows"] == 8
    assert result["repeated_rows"] == 3
    assert result["optimizer_updates"] == 1
    assert result["unique_supervised_tokens"] == 28
    assert result["sampled_supervised_tokens"] == sum(
        count * tokens for count, tokens in zip(result["exposure_counts"], targets, strict=True)
    )


def test_distributed_exposure_preserves_partial_accumulation_update():
    result = distributed_epoch_exposure(
        list(range(1, 13)), [1] * 12, batch_size=2, processes=2, accumulation=2, seed=42
    )
    assert result["sampled_rows"] == 12
    assert result["microbatches_per_rank"] == 3
    assert result["optimizer_updates"] == 2


def test_distributed_exposure_rejects_missing_targets():
    with pytest.raises(ValueError, match="align"):
        distributed_epoch_exposure([1], [], batch_size=2, processes=2, accumulation=2, seed=42)


@pytest.mark.parametrize("processes", [1, 2])
@pytest.mark.parametrize("size", [5, 12, 13])
def test_exposure_matches_actual_accelerate_loader(processes, size):
    from collections import Counter

    import torch
    from accelerate.data_loader import prepare_data_loader
    from torch.utils.data import DataLoader

    from invllava.train.sampler import ModalityLengthGroupedSampler

    lengths = [i + 1 if i % 3 else -(i + 1) for i in range(size)]
    observed = Counter()
    for rank in range(processes):
        sampler = ModalityLengthGroupedSampler(
            lengths, batch_size=2, group_count=processes * 2, seed=42
        )
        loader = DataLoader(list(range(size)), batch_size=2, sampler=sampler, drop_last=False)
        prepared = prepare_data_loader(
            loader,
            device=torch.device("cpu"),
            num_processes=processes,
            process_index=rank,
            split_batches=False,
            even_batches=True,
            put_on_device=False,
            rng_types=[],
        )
        for batch in prepared:
            observed.update(batch.tolist())
    result = distributed_epoch_exposure(
        lengths, [1] * size, batch_size=2, processes=processes, accumulation=2, seed=42
    )
    assert result["exposure_counts"] == [observed[i] for i in range(size)]
    if processes == 1:
        assert result["sampled_rows"] == size and result["repeated_rows"] == 0
