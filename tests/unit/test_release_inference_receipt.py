import json

import pytest
import torch

from invllava.model.types import ExpandedSequence, FusionState
from scripts.verify_release_inference import attention_context, without_visual_input, write_failure


@pytest.mark.parametrize("projected_tokens", [False, True])
def test_feature_intervention_removes_correct_interface_without_mutating_input(projected_tokens):
    state = FusionState(
        visual_features=torch.ones(1, 2, 4),
        text_mask=torch.tensor([[True, False]]),
        vision_mask=torch.tensor([[False, True]]),
    )
    original = ExpandedSequence(
        inputs_embeds=torch.ones(1, 2, 4),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        position_ids=torch.arange(2).unsqueeze(0),
        labels=None,
        fusion_state=state,
    )
    ablated = without_visual_input(original, projected_tokens=projected_tokens)
    assert not ablated.fusion_state.visual_features.any()
    assert torch.equal(ablated.inputs_embeds[:, 0], original.inputs_embeds[:, 0])
    assert bool(ablated.inputs_embeds[:, 1].any()) != projected_tokens
    assert original.inputs_embeds.all() and original.fusion_state.visual_features.all()
    assert torch.equal(original.attention_mask, ablated.attention_mask)
    assert torch.equal(original.position_ids, ablated.position_ids)


def test_failed_acceptance_preserves_both_unedited_predictions(tmp_path):
    output = tmp_path / "failed.json"
    details = {"sample_id": "fixture", "first_prediction": " A ", "second_prediction": "B"}
    write_failure(
        output, checkpoint_sha256="a" * 64, reason="greedy-repeat mismatch", details=details
    )
    value = json.loads(output.read_text())
    assert value["status"] == "failed"
    assert value["checkpoint_sha256"] == "a" * 64
    assert value["details"] == details
    assert len(value["verifier_sha256"]) == 64


def test_failure_does_not_overwrite_an_existing_receipt(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(FileExistsError):
        write_failure(output, checkpoint_sha256="a" * 64, reason="fixture", details={})
    assert output.read_text() == "preserve"


def enabled_backends():
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
    )


@pytest.mark.parametrize("kernel", ["auto", "flash", "math"])
def test_optional_kernel_is_scoped_and_restored_on_failure(kernel):
    before = enabled_backends()
    with pytest.raises(RuntimeError, match="fixture"):
        with attention_context(kernel):
            expected = {
                "auto": before,
                "flash": (True, False, False, False),
                "math": (False, True, False, False),
            }[kernel]
            assert enabled_backends() == expected
            raise RuntimeError("fixture")
    assert enabled_backends() == before


def test_unknown_kernel_rejected_without_changing_global_state():
    before = enabled_backends()
    with pytest.raises(ValueError, match="unsupported"):
        attention_context("unknown")
    assert enabled_backends() == before
