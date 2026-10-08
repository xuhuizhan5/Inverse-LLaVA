"""Precision, loading, casting, and resume contracts for unscaled Vicuna RoPE."""

import pytest
import torch

from invllava.model.llama.attention import RotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp32_policy_survives_module_cast_and_state_roundtrip(dtype):
    rotary = RotaryEmbedding(128, 10000.0, precision="float32")
    expected = rotary.inv_freq.clone()
    rotary.to(dtype=dtype)
    rotary.half().bfloat16().float()
    assert rotary.inv_freq.dtype == torch.float32
    assert torch.equal(rotary.inv_freq, expected)
    restored = RotaryEmbedding(128, 10000.0, precision="float32").to(dtype=dtype)
    restored.load_state_dict(rotary.state_dict())
    positions = torch.tensor([[0, 511, 1023, 2047]])
    for actual, reference in zip(restored(positions, dtype), rotary(positions, dtype), strict=True):
        assert actual.dtype == dtype
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_restoration_recovers_unrounded_values_and_preserves_existing_default(dtype):
    expected = RotaryEmbedding(128, 10000.0).inv_freq.clone()
    rotary = RotaryEmbedding(128, 10000.0).to(dtype=dtype)
    assert torch.equal(rotary.inv_freq, expected.to(dtype))
    assert not torch.equal(rotary.inv_freq.float(), expected)
    rotary.set_precision("float32")
    assert torch.equal(rotary.inv_freq, expected)
    rotary.to(dtype=dtype)
    assert torch.equal(rotary.inv_freq, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp32_resume_rejects_rounded_state_even_when_loading_non_strictly(dtype):
    target = RotaryEmbedding(128, 10000.0, precision="float32")
    rounded = RotaryEmbedding(128, 10000.0).to(dtype=dtype).state_dict()
    with pytest.raises(RuntimeError, match="cannot resume rounded"):
        target.load_state_dict(rounded, strict=False)


def test_fp32_policy_rejects_changed_frequencies_and_detects_external_recast():
    rotary = RotaryEmbedding(128, 10000.0)
    rotary.inv_freq[1] += 0.01
    with pytest.raises(ValueError, match="do not match unscaled"):
        rotary.set_precision("float32")
    rotary = RotaryEmbedding(128, 10000.0, precision="float32")
    rotary.inv_freq = rotary.inv_freq.bfloat16()
    with pytest.raises(RuntimeError, match="recast after loading"):
        rotary(torch.tensor([[0, 1]]), torch.bfloat16)


def test_fp32_rope_remains_float32_under_autocast():
    rotary = RotaryEmbedding(128, 10000.0, precision="float32")
    positions = torch.arange(2048).unsqueeze(0)
    expected = rotary(positions, torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = rotary(positions, torch.bfloat16)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))


@pytest.mark.parametrize("head_dim,theta", [(0, 10000), (3, 10000), (128, 0), (128, float("nan"))])
def test_invalid_rotary_geometry_rejected(head_dim, theta):
    with pytest.raises(ValueError, match="positive"):
        RotaryEmbedding(head_dim, theta)
