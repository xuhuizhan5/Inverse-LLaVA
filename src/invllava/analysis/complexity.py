from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FusionComplexity:
    parameters: int
    macs_per_forward: int
    concat_activation_elements: int
    projected_activation_elements: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def fusion_complexity(
    *,
    hidden_size: int,
    visual_size: int,
    sequence_length: int,
    target_output_sizes: tuple[int, ...],
    visual_input_size: int | None = None,
    fusion_layers: int = 1,
    operator: str = "concat",
    learned_scales: bool = True,
    visual_normalization: bool = True,
) -> FusionComplexity:
    if min(hidden_size, visual_size, sequence_length, fusion_layers) <= 0:
        raise ValueError("dimensions, sequence length, and layer count must be positive")
    if operator not in {"concat", "add", "gated"}:
        raise ValueError(f"unsupported fusion operator: {operator}")
    if not target_output_sizes or min(target_output_sizes) <= 0:
        raise ValueError("at least one positive attention target width is required")
    if visual_input_size is not None and visual_input_size <= 0:
        raise ValueError("visual input width must be positive")
    target_count = len(target_output_sizes)
    mapper = target_count * hidden_size * visual_size
    visual_reducer = (
        0 if visual_input_size in (None, visual_size) else int(visual_input_size) * visual_size
    )
    joint_size = 2 * visual_size if operator == "concat" else visual_size
    outputs = joint_size * sum(target_output_sizes)
    gate_weights = 2 * visual_size * visual_size if operator == "gated" else 0
    gate_bias = visual_size if operator == "gated" else 0
    scales = target_count if learned_scales else 0
    visual_norm = target_count * visual_size if visual_normalization else 0
    per_layer_parameters = (
        mapper + visual_reducer + outputs + gate_weights + gate_bias + scales + visual_norm
    )
    # Bias addition, sigmoid, normalization, and scalar products are elementwise
    # work; this count covers dense projection multiply-accumulates only.
    # The gate's weights are shared, but _joint evaluates them separately for
    # each target-specific text/visual stream.
    per_layer_macs = sequence_length * (
        mapper + outputs + target_count * gate_weights + visual_reducer
    )
    return FusionComplexity(
        parameters=fusion_layers * per_layer_parameters,
        macs_per_forward=fusion_layers * per_layer_macs,
        concat_activation_elements=(
            fusion_layers * sequence_length * joint_size if operator == "concat" else 0
        ),
        projected_activation_elements=fusion_layers * sequence_length * sum(target_output_sizes),
    )


def llava_projector_complexity(
    hidden_size: int,
    visual_size: int,
    patches: int,
    *,
    kind: str = "mlp2x_gelu",
) -> dict[str, int]:
    if min(hidden_size, visual_size, patches) <= 0:
        raise ValueError("dimensions and patch count must be positive")
    if kind == "linear":
        weight_terms = hidden_size * visual_size
        biases = hidden_size
    elif kind == "mlp2x_gelu":
        weight_terms = visual_size * hidden_size + hidden_size * hidden_size
        biases = 2 * hidden_size
    else:
        raise ValueError(f"unsupported LLaVA projector kind: {kind}")
    return {
        "parameters": weight_terms + biases,
        "macs_per_image": patches * weight_terms,
    }
