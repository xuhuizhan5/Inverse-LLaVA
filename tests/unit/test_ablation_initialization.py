import json

import pytest
import torch
from torch import nn

from invllava.config.loader import ConfigRepository
from invllava.model.fusion import FusionBlock
from invllava.train.checkpoint import checkpoint_parameter_names, load_trainable_weights
from scripts.audit_ablation_initialization import (
    align_common_initialization,
    save_initialization,
    validate_model_contract,
)


class InitializationFixture(nn.Module):
    def __init__(self, layer=0, operator="concat", visual_size=2):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False).requires_grad_(False)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([nn.Module() for _ in range(3)])
        attention = nn.Module()
        attention.fusion = FusionBlock(
            hidden_size=4,
            visual_size=visual_size,
            target_sizes={"q": 4},
            targets=("q",),
            operator=operator,
        )
        self.language_model.model.layers[layer].self_attn = attention
        self.lora_a = nn.Linear(4, 2, bias=False)


def delta(model):
    names = checkpoint_parameter_names(model)
    return {
        name: value.detach().clone() for name, value in model.named_parameters() if name in names
    }


@pytest.mark.parametrize("operator", ["add", "gated"])
def test_operator_initialization_matches_shared_tensors_without_replacing_base(operator):
    torch.manual_seed(42)
    reference = delta(InitializationFixture())
    torch.manual_seed(42)
    candidate = InitializationFixture(operator=operator)
    base = candidate.base.weight.detach().clone()
    output_name = "language_model.model.layers.0.self_attn.fusion.output.q.weight"
    output = delta(candidate)[output_name]
    report = align_common_initialization(candidate, reference, reference_layer=0, candidate_layer=0)
    state = delta(candidate)
    for target, source in report["copied_from_reference"].items():
        assert torch.equal(state[target], reference[source])
    assert torch.equal(state[output_name], output)
    assert torch.equal(candidate.base.weight, base)
    assert output_name in report["retained_candidate_initializers"]


def test_depth_initialization_moves_fusion_and_preserves_absolute_lora_identity():
    reference = delta(InitializationFixture())
    candidate = InitializationFixture(layer=2)
    report = align_common_initialization(candidate, reference, reference_layer=0, candidate_layer=2)
    assert report["copied_from_reference"]["lora_a.weight"] == "lora_a.weight"
    state = delta(candidate)
    for target, source in report["copied_from_reference"].items():
        assert torch.equal(state[target], reference[source])
    assert not report["retained_candidate_initializers"]


def test_width_initialization_preserves_width_specific_tensors_and_matches_lora():
    reference = delta(InitializationFixture())
    candidate = InitializationFixture(visual_size=4)
    before = delta(candidate)
    with pytest.raises(ValueError, match="shape mismatch"):
        align_common_initialization(candidate, reference, reference_layer=0, candidate_layer=0)
    assert all(torch.equal(before[name], value) for name, value in delta(candidate).items())
    report = align_common_initialization(
        candidate,
        reference,
        reference_layer=0,
        candidate_layer=0,
        allow_feature_width_change=True,
    )
    after = delta(candidate)
    assert report["copied_from_reference"]["lora_a.weight"] == "lora_a.weight"
    assert len(report["retained_candidate_initializers"]) == 3
    for name, source in report["copied_from_reference"].items():
        assert torch.equal(after[name], reference[source])
    for name in report["retained_candidate_initializers"]:
        assert torch.equal(after[name], before[name])


def test_width_contract_requires_explicit_scope_and_same_encoder_backbone():
    from pathlib import Path

    import yaml

    from invllava.config.schema import ModelSpec

    repository = ConfigRepository("configs")
    reference = repository.resolve("configs/experiment/canonical_5pct_calibration.yaml").model
    hd = ModelSpec.model_validate(
        yaml.safe_load(Path("configs/model/inverse_vicuna7b_hd.yaml").read_text())
    )
    with pytest.raises(ValueError, match="CLIP feature-layer"):
        validate_model_contract(reference, hd)
    validate_model_contract(reference, hd, allow_feature_width_change=True)
    larger = repository.resolve("configs/experiment/scaling_13b_5pct.yaml").model
    with pytest.raises(ValueError, match="identical language"):
        validate_model_contract(reference, larger, allow_feature_width_change=True)
    resized = hd.model_copy(update={"vision": hd.vision.model_copy(update={"image_size": 448})})
    with pytest.raises(ValueError, match="CLIP feature-layer"):
        validate_model_contract(reference, resized, allow_feature_width_change=True)
    invalid = hd.model_copy(update={"vision": hd.vision.model_copy(update={"feature_dim": 1536})})
    with pytest.raises(ValueError, match="same encoder"):
        validate_model_contract(reference, invalid, allow_feature_width_change=True)


def test_initialization_rejects_unexplained_shape_or_missing_tensor():
    candidate = InitializationFixture()
    reference = delta(candidate)
    reference["lora_a.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        align_common_initialization(candidate, reference, reference_layer=0, candidate_layer=0)
    reference.pop("lora_a.weight")
    with pytest.raises(ValueError, match="LoRA initialization targets differ"):
        align_common_initialization(candidate, reference, reference_layer=0, candidate_layer=0)


class ProjectorFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False).requires_grad_(False)
        self.lora_a = nn.Linear(4, 2, bias=False)
        self.multimodal_projector = nn.Sequential(nn.Linear(2, 4), nn.GELU(), nn.Linear(4, 4))


@pytest.mark.parametrize("reverse", [False, True])
def test_cross_interface_copies_all_lora_and_preserves_interface_and_base(reverse):
    inverse, conventional = InitializationFixture(), ProjectorFixture()
    source, candidate = (conventional, inverse) if reverse else (inverse, conventional)
    before = delta(candidate)
    base = candidate.base.weight.detach().clone()
    report = align_common_initialization(
        candidate,
        delta(source),
        reference_layer=None if reverse else 0,
        candidate_layer=0 if reverse else None,
    )
    actual = delta(candidate)
    assert report["copied_from_reference"] == {"lora_a.weight": "lora_a.weight"}
    assert torch.equal(actual["lora_a.weight"], delta(source)["lora_a.weight"])
    assert report["retained_candidate_initializers"]
    for name in report["retained_candidate_initializers"]:
        assert torch.equal(actual[name], before[name])
    assert torch.equal(candidate.base.weight, base)


def test_cross_interface_rejects_incomplete_lora_before_modifying_any_weights():
    candidate = ProjectorFixture()
    before = delta(candidate)
    source = delta(InitializationFixture())
    source["lora_b.weight"] = torch.ones(4, 2)
    with pytest.raises(ValueError, match="LoRA initialization targets differ"):
        align_common_initialization(candidate, source, reference_layer=0, candidate_layer=None)
    assert all(torch.equal(before[name], value) for name, value in delta(candidate).items())


def test_cross_interface_rejects_unknown_trainable_tensor():
    candidate = ProjectorFixture()
    candidate.unexplained = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="unexplained candidate-only"):
        align_common_initialization(
            candidate, delta(InitializationFixture()), reference_layer=0, candidate_layer=None
        )


def test_projector_feature_pair_matches_interface_initialization():
    reference, candidate = ProjectorFixture(), ProjectorFixture()
    report = align_common_initialization(
        candidate, delta(reference), reference_layer=None, candidate_layer=None
    )
    assert not report["retained_candidate_initializers"]
    expected = delta(reference)
    assert all(torch.equal(value, expected[name]) for name, value in delta(candidate).items())


def test_model_contract_rejects_aligned_projector_and_accepts_feature_selection():
    repository = ConfigRepository("configs")
    inverse = repository.resolve("configs/experiment/canonical_5pct_calibration.yaml").model
    conventional = repository.resolve("configs/experiment/controlled_llava_lora.yaml").model
    with pytest.raises(ValueError, match="random, untrained projector"):
        validate_model_contract(inverse, conventional)
    random = conventional.model_copy(
        update={
            "projector": conventional.projector.model_copy(
                update={"initialization": "random", "initial_checkpoint_id": None}
            )
        }
    )
    validate_model_contract(inverse, random)
    changed_width = random.model_copy(
        update={"vision": random.vision.model_copy(update={"feature_dim": 2048})}
    )
    with pytest.raises(ValueError, match="CLIP feature-layer"):
        validate_model_contract(inverse, changed_width)


def test_projector_untrained_delta_round_trip(tmp_path):
    model = ProjectorFixture()
    expected = delta(model)
    recipe = tmp_path / "input.yaml"
    recipe.write_text("id: fixture\n")
    save_initialization(tmp_path / "initialization", model, str(recipe), {"seed": 42})
    reloaded = ProjectorFixture()
    load_trainable_weights(tmp_path / "initialization", reloaded)
    assert all(torch.equal(value, delta(reloaded)[name]) for name, value in expected.items())


def test_untrained_delta_round_trip_and_immutability(tmp_path):
    model = InitializationFixture()
    expected = delta(model)
    recipe = tmp_path / "input.yaml"
    recipe.write_text("id: fixture\n")
    output = tmp_path / "initialization"
    digest = save_initialization(output, model, str(recipe), {"seed": 42})
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["trained_updates"] == 0
    assert metadata["artifact_kind"] == "untrained_initialization"
    assert digest in (output / "experiment.yaml").read_text()
    assert not (output / "training_state.pt").exists()
    reloaded = InitializationFixture()
    load_trainable_weights(output, reloaded)
    for name, value in delta(reloaded).items():
        assert torch.equal(value, expected[name])
    with pytest.raises(FileExistsError):
        save_initialization(output, model, str(recipe), {})


def test_exact_saved_reference_replaces_seeded_values_and_rejects_trained_state(tmp_path):
    from scripts.audit_ablation_initialization import load_initialization_reference

    resolved = ConfigRepository("configs").resolve(
        "configs/experiment/canonical_5pct_calibration.yaml"
    )
    reference = InitializationFixture()
    recipe = tmp_path / "source.yaml"
    recipe.write_text("id: fixture\n")
    target = tmp_path / "anchor"
    save_initialization(
        target,
        reference,
        str(recipe),
        {
            "seed": resolved.training.seed,
            "resolved_model": resolved.model.model_dump(mode="json"),
        },
    )
    changed = InitializationFixture()
    for parameter in changed.parameters():
        with torch.no_grad():
            parameter.zero_()
    receipt = load_initialization_reference(target, changed, resolved)
    assert len(receipt["delta_sha256"]) == 64
    assert all(torch.equal(value, delta(changed)[name]) for name, value in delta(reference).items())
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["trained_updates"] = 1
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="explicitly untrained"):
        load_initialization_reference(target, changed, resolved)
