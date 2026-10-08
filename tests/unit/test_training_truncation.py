import pytest

from scripts.audit_training_truncation import supervision_summary


def test_expansion_can_remove_all_supervision():
    result = supervision_summary(
        [1, -200, 2, 3], [-100, -100, 2, 3], image_patch_counts=[4], max_length=5
    )
    assert result["raw_targets"] == result["collated_targets"] == 2
    assert result["surviving_targets"] == 0
    assert result["targets_lost_after_expansion"] == 2
    assert result["truncated_at_collation"] == 0
    assert result["truncated_after_expansion"] == 1


def test_separate_collation_and_expansion_losses():
    result = supervision_summary(
        [1, -200, 2, 3, 4, 5], [-100, -100, 2, 3, 4, 5], image_patch_counts=[2], max_length=5
    )
    assert result["targets_lost_at_collation"] == 1
    assert result["targets_lost_after_expansion"] == 1
    assert result["surviving_targets"] == 2


def test_text_only_and_first_position_causal_shift():
    result = supervision_summary([1, 2, 3], [1, 2, 3], image_patch_counts=[], max_length=3)
    assert result["raw_targets"] == result["surviving_targets"] == 2
    assert result["truncated_after_expansion"] == 0


def test_all_masked_context_is_audited():
    result = supervision_summary([1, 2], [-100, -100], image_patch_counts=[], max_length=3)
    assert result["raw_zero_targets"] == result["surviving_zero_targets"] == 1


def test_lost_image_placeholder_is_a_contract_error():
    with pytest.raises(ValueError, match="placeholder"):
        supervision_summary([1, 2, -200], [-100, 2, -100], image_patch_counts=[4], max_length=2)
