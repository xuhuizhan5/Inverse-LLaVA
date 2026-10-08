import torch
from torch import nn

from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.modeling import LLaVAReferenceForConditionalGeneration
from invllava.model.projector import MLP2xGELUProjector
from invllava.model.projector_checkpoint import extract_projector_state


def test_extracts_original_llava_projector_names() -> None:
    state = {
        "model.mm_projector.0.weight": torch.randn(8, 4),
        "model.mm_projector.0.bias": torch.randn(8),
        "model.mm_projector.2.weight": torch.randn(8, 8),
        "model.mm_projector.2.bias": torch.randn(8),
        "ignored.weight": torch.randn(1),
    }
    extracted = extract_projector_state(state)
    assert set(extracted) == {
        "linear_1.weight",
        "linear_1.bias",
        "linear_2.weight",
        "linear_2.bias",
    }


def test_projector_shape() -> None:
    projector = MLP2xGELUProjector(4, 8)
    assert projector(torch.randn(2, 3, 4)).shape == (2, 3, 8)


def test_reference_inserts_projected_features_as_language_inputs() -> None:
    architecture = LlamaArchitecture(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    language = InverseLlamaForCausalLM(architecture)
    model = LLaVAReferenceForConditionalGeneration(
        language,
        nn.Identity(),
        nn.Identity(),
        hidden_size=8,
        max_length=16,
    )
    patches = torch.randn(2, 8)
    expanded = model.prepare(
        input_ids=torch.tensor([[1, -200, 3]]),
        image_features=[[patches]],
    )
    torch.testing.assert_close(expanded.inputs_embeds[0, 1:3], patches)
    torch.testing.assert_close(expanded.inputs_embeds[0, 0], language.model.embed_tokens.weight[1])
    torch.testing.assert_close(expanded.inputs_embeds[0, 3], language.model.embed_tokens.weight[3])
