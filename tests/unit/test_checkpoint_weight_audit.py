"""Synthetic stored-coordinate comparisons, unrelated to benchmark evidence."""

import math
import warnings

import pytest
import torch
from safetensors.torch import save_file

from scripts.compare_checkpoint_weights import compare_files, tensor_statistics


def test_known_changes_and_chunking():
    left, right = torch.tensor([3.0, 4.0, 0.0]), torch.tensor([3.0, 5.0, 2.0])
    result = tensor_statistics(left, right, chunk_size=1)
    assert result == tensor_statistics(left, right, chunk_size=8)
    assert result["changed_elements"] == 2
    assert result["reference_squared_norm"] == 25
    assert result["change_squared_norm"] == 5
    assert result["maximum_absolute_change"] == 2


def test_bfloat16_stored_values():
    left = torch.tensor([1.0], dtype=torch.bfloat16)
    right = torch.tensor([1.0078125], dtype=torch.bfloat16)
    assert tensor_statistics(left, right)["change_squared_norm"] == 0.0078125**2


def test_inspection_detaches_trainable_tensors():
    left = torch.tensor([1.0], requires_grad=True)
    right = torch.tensor([2.0], requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = tensor_statistics(left, right)
    assert not caught
    assert result["change_squared_norm"] == 1
    assert left.grad is None and right.grad is None


@pytest.mark.parametrize(
    "right", [torch.ones(3), torch.ones(2, dtype=torch.float64), torch.tensor([float("nan"), 0.0])]
)
def test_reject_incompatible_or_nonfinite(right):
    with pytest.raises(ValueError):
        tensor_statistics(torch.ones(2), right)


def test_file_comparison_and_zero_reference(tmp_path):
    left, right = tmp_path / "left.safetensors", tmp_path / "right.safetensors"
    save_file(
        {
            "model.lora_a.weight": torch.tensor([3.0, 4.0]),
            "model.fusion.scales.q": torch.tensor(0.0),
        },
        left,
    )
    save_file(
        {
            "model.lora_a.weight": torch.tensor([3.0, 5.0]),
            "model.fusion.scales.q": torch.tensor(1.0),
        },
        right,
    )
    result = compare_files(left, right)
    assert result["groups"]["lora_a"]["relative_l2_change"] == pytest.approx(0.2)
    assert result["groups"]["fusion.scales"]["relative_l2_change"] is None
    assert result["checkpoint_sha256"]["reference"] != result["checkpoint_sha256"]["candidate"]
    assert compare_files(left, left)["groups"]["lora_a"]["changed_fraction"] == 0


def test_reject_different_names(tmp_path):
    left, right = tmp_path / "left.safetensors", tmp_path / "right.safetensors"
    save_file({"a": torch.ones(1)}, left)
    save_file({"b": torch.ones(1)}, right)
    with pytest.raises(ValueError, match="names"):
        compare_files(left, right)


def test_scalar_and_empty_tensor():
    scalar = tensor_statistics(torch.tensor(1.0), torch.tensor(1.0))
    assert scalar["reference_values"] == [1.0]
    assert scalar["changed_elements"] == 0
    empty = tensor_statistics(torch.empty(0), torch.empty(0))
    assert empty["elements"] == 0 and math.isfinite(empty["change_squared_norm"])
