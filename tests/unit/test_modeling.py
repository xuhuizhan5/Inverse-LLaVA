import pytest
import torch

from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.modeling import (
    InverseLLaVAForConditionalGeneration,
    LLaVAReferenceForConditionalGeneration,
    _zero_parameter_dependency,
)


def test_zero_dependency_marks_trainable_projector_parameters_used() -> None:
    projector = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.GELU(),
        torch.nn.Linear(5, 4),
    )
    reference = torch.randn(2, 4, requires_grad=True)
    output = reference.sum() + _zero_parameter_dependency(projector, reference)
    output.backward()

    assert all(parameter.grad is not None for parameter in projector.parameters())
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in projector.parameters())


def test_forward_allows_masked_context_row_beside_supervised_row() -> None:
    architecture = LlamaArchitecture(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = InverseLLaVAForConditionalGeneration(
        InverseLlamaForCausalLM(architecture, attention_backend="eager"),
        torch.nn.Identity(),
        visual_feature_dim=8,
        max_length=8,
    )
    input_ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        labels=torch.tensor([[-100, -100, -100, -100], [-100, 7, 8, 9]]),
        image_features=[[], []],
    )

    assert output.loss is not None
    assert torch.isfinite(output.loss)
    assert output.supervised_token_count == 3
    assert output.expanded_token_count == 8


@pytest.mark.parametrize("architecture_kind", ["inverse", "projector"])
def test_training_counts_follow_expanded_labels_without_changing_gradients(architecture_kind):
    architecture = LlamaArchitecture(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    language = InverseLlamaForCausalLM(architecture, attention_backend="eager")
    if architecture_kind == "inverse":
        model = InverseLLaVAForConditionalGeneration(
            language, torch.nn.Identity(), visual_feature_dim=8, max_length=6
        )
    else:
        model = LLaVAReferenceForConditionalGeneration(
            language, torch.nn.Identity(), torch.nn.Identity(), hidden_size=8, max_length=6
        )
    batch = {
        "input_ids": torch.tensor([[1, -200, 4, 5, 6], [1, 4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool),
        "labels": torch.tensor([[-100, -100, 4, 5, 6], [-100, 4, 5, -100, -100]]),
        "image_features": [[torch.randn(4, 8)], []],
    }
    expanded = model.prepare(**batch)
    direct = language(
        inputs_embeds=expanded.inputs_embeds,
        attention_mask=expanded.attention_mask,
        position_ids=expanded.position_ids,
        labels=expanded.labels,
        fusion_state=expanded.fusion_state if architecture_kind == "inverse" else None,
    )
    output = model(**batch)
    assert batch["labels"].ne(-100).sum() == 5
    assert output.supervised_token_count == 3
    assert output.expanded_token_count == 9
    torch.testing.assert_close(output.logits, direct.logits, rtol=0, atol=0)
    torch.testing.assert_close(output.loss, direct.loss, rtol=0, atol=0)
    parameter = language.lm_head.weight
    observed = torch.autograd.grad(output.loss, parameter)[0]
    expected = torch.autograd.grad(direct.loss, parameter)[0]
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    inference = model(**{key: value for key, value in batch.items() if key != "labels"})
    assert inference.supervised_token_count is None
    assert inference.expanded_token_count is None
