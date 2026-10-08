from invllava.train.sampler import (
    ModalityLengthGroupedSampler,
    modality_length_grouped_indices,
)


def test_modality_grouping_is_complete_deterministic_and_epoch_seeded() -> None:
    lengths = [5, 20, 8, 14, -3, -17, -6, -11, 9, 13, -7, -15]
    first = modality_length_grouped_indices(lengths, batch_size=2, group_count=2, seed=42, epoch=0)
    repeated = modality_length_grouped_indices(
        lengths, batch_size=2, group_count=2, seed=42, epoch=0
    )
    next_epoch = modality_length_grouped_indices(
        lengths, batch_size=2, group_count=2, seed=42, epoch=1
    )

    assert first == repeated
    assert sorted(first) == list(range(len(lengths)))
    assert first != next_epoch
    # Every complete four-example megabatch contains one modality.
    for offset in range(0, 8, 4):
        assert len({lengths[index] > 0 for index in first[offset : offset + 4]}) == 1


def test_grouped_sampler_changes_order_by_epoch_without_global_rng() -> None:
    sampler = ModalityLengthGroupedSampler(
        [3, 8, 5, 12, -4, -9, -7, -11], batch_size=2, group_count=2, seed=7
    )
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(1)
    assert sorted(sampler) == list(range(8))
    assert list(sampler) != first


def test_grouping_rejects_ambiguous_zero_length() -> None:
    try:
        modality_length_grouped_indices([2, 0], batch_size=1, group_count=1, seed=1, epoch=0)
    except ValueError as error:
        assert "non-zero" in str(error)
    else:
        raise AssertionError("zero length should fail")
