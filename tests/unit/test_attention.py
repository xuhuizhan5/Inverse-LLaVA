import torch

from invllava.model.llama.attention import InverseLlamaAttention
from invllava.model.llama.configuration import LlamaArchitecture


def _attention_pair() -> tuple[InverseLlamaAttention, InverseLlamaAttention]:
    config = LlamaArchitecture(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    eager = InverseLlamaAttention(config, backend="eager").eval()
    sdpa = InverseLlamaAttention(config, backend="sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())
    return eager, sdpa


def test_rotary_frequency_buffer_uses_vicuna_checkpoint_name() -> None:
    attention, _ = _attention_pair()
    state = attention.state_dict()
    assert "rotary_emb.inv_freq" in state
    assert state["rotary_emb.inv_freq"].shape == (2,)


def test_sdpa_causal_fast_path_matches_explicit_eager_mask() -> None:
    torch.manual_seed(13)
    eager, sdpa = _attention_pair()
    hidden = torch.randn(2, 5, 16)
    positions = torch.arange(5).unsqueeze(0).expand(2, -1)
    explicit = torch.ones((2, 5), dtype=torch.bool)

    expected, _ = eager(
        hidden,
        position_ids=positions,
        attention_mask=explicit,
        fusion_state=None,
    )
    actual, _ = sdpa(
        hidden,
        position_ids=positions,
        attention_mask=None,
        fusion_state=None,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_sdpa_boolean_padding_mask_matches_eager() -> None:
    torch.manual_seed(17)
    eager, sdpa = _attention_pair()
    hidden = torch.randn(2, 5, 16)
    positions = torch.tensor([[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]])
    mask = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])

    expected, _ = eager(
        hidden,
        position_ids=positions,
        attention_mask=mask,
        fusion_state=None,
    )
    actual, _ = sdpa(
        hidden,
        position_ids=positions,
        attention_mask=mask,
        fusion_state=None,
    )

    # Fully padded query positions have no valid attention distribution and may
    # differ across kernels. Valid queries must remain numerically equivalent.
    torch.testing.assert_close(actual[mask], expected[mask], atol=1e-5, rtol=1e-5)
