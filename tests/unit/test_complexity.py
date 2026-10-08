import pytest

from invllava.analysis.complexity import fusion_complexity, llava_projector_complexity


def test_recorded_width_study_interface_counts() -> None:
    # Pinned Vicuna widths; three full-width Q/K/V branches, one fusion layer.
    for hidden, visual, parameters, macs in (
        (4096, 1024, 37_751_811, 37_748_736),
        (4096, 2048, 75_503_619, 75_497_472),
        (5120, 1024, 47_188_995, 47_185_920),
    ):
        result = fusion_complexity(
            hidden_size=hidden,
            visual_size=visual,
            sequence_length=1,
            target_output_sizes=(hidden,) * 3,
        )
        assert result.parameters == parameters
        assert result.macs_per_forward == macs


def test_canonical_concat_parameter_formula() -> None:
    hidden = 4096
    visual = 1024
    result = fusion_complexity(
        hidden_size=hidden,
        visual_size=visual,
        sequence_length=2048,
        target_output_sizes=(hidden, hidden, hidden),
    )
    assert result.parameters == 9 * hidden * visual + 3 * visual + 3
    assert result.macs_per_forward == 2048 * 9 * hidden * visual


def test_reduced_width_counts_explicit_visual_reducer() -> None:
    result = fusion_complexity(
        hidden_size=8,
        visual_size=2,
        visual_input_size=4,
        sequence_length=5,
        target_output_sizes=(8, 8, 8),
    )
    # Three text mappers, shared reducer, three concat projections, norms, scales.
    assert result.parameters == (3 * 8 * 2) + (4 * 2) + (4 * 24) + (3 * 2) + 3
    assert result.macs_per_forward == 5 * ((3 * 8 * 2) + (4 * 2) + (4 * 24))


def test_llava_mlp_projector_counts_both_linear_layers_and_biases() -> None:
    result = llava_projector_complexity(8, 4, 3)
    assert result["parameters"] == (4 * 8) + (8 * 8) + (2 * 8)
    assert result["macs_per_image"] == 3 * ((4 * 8) + (8 * 8))


def test_paper_llava_mlp_count() -> None:
    result = llava_projector_complexity(4096, 1024, 576)
    assert result == {"parameters": 20_979_712, "macs_per_image": 12_079_595_520}


def test_gate_bias_is_a_parameter_without_being_a_multiply_accumulate() -> None:
    result = fusion_complexity(
        hidden_size=8,
        visual_size=4,
        sequence_length=5,
        target_output_sizes=(8, 8, 8),
        operator="gated",
    )
    assert result.parameters == 3 * 8 * 4 + 4 * 24 + 2 * 4 * 4 + 4 + 3 + 3 * 4
    assert result.macs_per_forward == 5 * (3 * 8 * 4 + 4 * 24 + 3 * 2 * 4 * 4)


@pytest.mark.parametrize("operator", ["concat", "add", "gated"])
@pytest.mark.parametrize("widths", [(8, 8, 8), (8, 4, 4)])
def test_complexity_matches_executed_linear_layers(operator, widths) -> None:
    """Count actual calls, including repeated use of the shared gate."""
    import torch

    from invllava.model.fusion import FusionBlock
    from invllava.model.types import FusionState

    block = FusionBlock(
        hidden_size=8,
        visual_size=4,
        target_sizes=dict(zip(("q", "k", "v"), widths, strict=True)),
        operator=operator,
    )
    calls = []

    def count_dense_work(module, inputs, output):
        calls.append(output.numel() * module.in_features)

    handles = [
        module.register_forward_hook(count_dense_work)
        for module in block.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    text = torch.tensor([[True, True, False, False, True]])
    try:
        block(torch.ones(1, 5, 8), FusionState(torch.ones(1, 5, 4), text, ~text))
    finally:
        for handle in handles:
            handle.remove()
    expected = fusion_complexity(
        hidden_size=8,
        visual_size=4,
        sequence_length=5,
        target_output_sizes=widths,
        operator=operator,
    )
    assert expected.parameters == block.extra_parameters()
    assert expected.macs_per_forward == sum(calls)


@pytest.mark.parametrize(
    "override",
    [
        {"operator": "unknown"},
        {"target_output_sizes": ()},
        {"target_output_sizes": (8, -1)},
        {"visual_input_size": 0},
    ],
)
def test_invalid_fusion_complexity_is_rejected(override) -> None:
    arguments = dict(hidden_size=8, visual_size=4, sequence_length=5, target_output_sizes=(8, 8, 8))
    with pytest.raises(ValueError):
        fusion_complexity(**(arguments | override))
