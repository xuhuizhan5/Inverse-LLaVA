"""Differential contract against the upstream Llama implementation."""

import pytest
import torch

from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM


def test_native_llama_matches_transformers_without_fusion() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    torch.manual_seed(2026)
    reference = transformers.LlamaForCausalLM(config).eval()
    native = InverseLlamaForCausalLM(
        LlamaArchitecture.from_hf(config), attention_backend="eager"
    ).eval()
    incompatible = native.load_state_dict(reference.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "model.layers.0.self_attn.rotary_emb.inv_freq",
        "model.layers.1.self_attn.rotary_emb.inv_freq",
    }
    assert not incompatible.unexpected_keys

    input_ids = torch.tensor([[1, 7, 9, 11, 13], [1, 5, 6, 8, 10]])
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        actual = native(input_ids=input_ids, attention_mask=attention_mask)
    torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pinned_original_llama_low_precision_with_long_positions(dtype) -> None:
    """Exercise the original cached RoPE against the explicit FP32 native policy.

    Run in the isolated Transformers 4.37.2 reference environment. The ordinary
    dependency environment retains the contemporary FP32 parity test above.
    """

    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.37.2":
        pytest.skip("requires the isolated original-reader Transformers 4.37.2 environment")
    config = transformers.LlamaConfig(
        vocab_size=67,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        pad_token_id=0,
    )
    torch.manual_seed(2026)
    reference = transformers.LlamaForCausalLM._from_config(
        config, attn_implementation="eager"
    ).eval()
    assert type(reference.model.layers[0].self_attn).__name__ == "LlamaAttention"
    native = InverseLlamaForCausalLM(
        LlamaArchitecture.from_hf(config), attention_backend="eager"
    ).eval()
    incompatible = native.load_state_dict(reference.state_dict(), strict=False)
    assert incompatible.missing_keys == ["model.layers.0.self_attn.rotary_emb.inv_freq"]
    assert not incompatible.unexpected_keys
    reference.to(dtype=dtype)
    native.to(dtype=dtype)
    rotary = native.model.layers[0].self_attn.rotary_emb
    rotary.set_precision("float32")
    native.to(dtype=dtype)
    input_ids = torch.randint(3, config.vocab_size, (1, 2048))
    with torch.inference_mode():
        expected = reference(input_ids=input_ids, use_cache=False).logits.float()
        actual = native(input_ids=input_ids, use_cache=False).logits.float()
    # Both readers execute the same eager low-precision linear operations.
    # Retain a tight absolute bound for possible elementwise rounding.
    tolerance = 5e-4 if dtype == torch.float16 else 4e-3
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
