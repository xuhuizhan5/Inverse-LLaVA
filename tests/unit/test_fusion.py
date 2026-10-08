import torch

from invllava.model.fusion import FusionBlock
from invllava.model.types import FusionState


def state(batch: int = 2, sequence: int = 5, visual: int = 4) -> FusionState:
    text_mask = torch.tensor([[1, 1, 0, 0, 1]] * batch, dtype=torch.bool)
    vision_mask = ~text_mask
    features = torch.randn(batch, sequence, visual) * vision_mask.unsqueeze(-1)
    return FusionState(features, text_mask, vision_mask)


def test_concat_fusion_shapes_and_gradients() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    updates = block(hidden, state())
    assert set(updates) == {"q", "k", "v"}
    assert updates["q"].shape == hidden.shape
    sum(value.sum() for value in updates.values()).backward()
    assert all(block.text_to_vision[target].weight.grad is not None for target in block.targets)


def test_frozen_mapper_stays_frozen() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
        mapper_trainable=False,
    )
    assert not any(parameter.requires_grad for parameter in block.text_to_vision.parameters())


def test_concat_equals_separate_modality_maps_in_values_and_gradients() -> None:
    """Bind the manuscript's linear decomposition to the executable branch."""
    torch.manual_seed(2026)
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
        parameter_dtype=torch.float64,
    )
    hidden = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    inputs = state()
    actual = block(hidden, inputs)
    expected = {}
    for target in block.targets:
        text = block.text_to_vision[target](hidden) * inputs.text_mask.unsqueeze(-1)
        visual = block.visual_norm[target](inputs.visual_features)
        visual = visual * inputs.vision_mask.unsqueeze(-1)
        text_map, visual_map = block.output[target].weight.split(4, dim=1)
        expected[target] = block.scales[target] * (
            text @ text_map.T + visual.to(text.dtype) @ visual_map.T
        )
        torch.testing.assert_close(actual[target], expected[target], rtol=1e-8, atol=1e-10)
    variables = (hidden, *block.parameters())
    actual_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in actual.values()), variables
    )
    expected_gradients = torch.autograd.grad(
        sum(value.square().sum() for value in expected.values()), variables
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-8, atol=1e-10)


def test_complete_fusion_branch_can_be_frozen() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
    )
    block.requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in block.parameters())


def test_target_mappers_and_visual_norms_are_independent() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
    )
    assert block.text_to_vision["q"].weight is not block.text_to_vision["k"].weight
    assert block.visual_norm["q"].weight is not block.visual_norm["v"].weight


def test_factorized_mapper_keeps_native_visual_output_width() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        mapper_rank=2,
        target_sizes={"q": 8, "k": 8, "v": 8},
    )
    mapper = block.text_to_vision["q"]
    assert isinstance(mapper, torch.nn.Sequential)
    assert mapper[0].in_features == 8
    assert mapper[0].out_features == 2
    assert mapper[1].in_features == 2
    assert mapper[1].out_features == 4
    assert block.output["q"].in_features == 8


def test_requested_parameter_dtype_is_used_during_construction() -> None:
    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes={"q": 8, "k": 8, "v": 8},
        parameter_dtype=torch.bfloat16,
    )
    assert {parameter.dtype for parameter in block.parameters()} == {torch.bfloat16}
