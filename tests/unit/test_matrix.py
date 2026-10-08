import json
from pathlib import Path

import pytest
import yaml

from invllava.config.loader import ConfigRepository
from invllava.config.matrix import materialize_experiment_matrix, set_config_path


def test_matrix_patch_rejects_unknown_top_level_paths() -> None:
    document = {"id": "fixture", "overrides": {}}
    with pytest.raises(ValueError, match="unknown path"):
        set_config_path(document, "unknown.value", 1)
    set_config_path(document, "overrides.model.fusion.layers", [0, 8])
    assert document["overrides"] == {"model": {"fusion": {"layers": [0, 8]}}}


def test_language_size_example_matches_qualified_microbatch_partition() -> None:
    repository = ConfigRepository("configs")
    larger = repository.resolve("configs/experiment/scaling_13b_5pct.yaml")
    control = repository.resolve(
        "configs/experiment/canonical_5pct_calibration.yaml", microbatch_size=16
    )
    assert larger.training == control.training
    assert larger.data == control.data
    assert larger.runtime == control.runtime
    assert larger.model.vision == control.model.vision
    assert larger.model.fusion == control.model.fusion
    assert larger.model.adaptation == control.model.adaptation
    assert larger.training.per_device_batch_size == 16
    assert larger.training.gradient_accumulation_steps == 4
    assert larger.runtime.num_processes == 2
    assert larger.model.language.hidden_size == 5120
    assert control.model.language.hidden_size == 4096


@pytest.mark.parametrize(
    ("filename", "names", "fraction"),
    [
        ("width_capacity_5pct_matrix", {"seven-billion", "hd", "thirteen-billion"}, 0.05),
        ("width_10pct_matrix", {"standard", "hd"}, 0.1),
    ],
)
def test_width_capacity_studies_match_data_schedule_and_adaptation(
    tmp_path: Path, filename: str, names: set[str], fraction: float
) -> None:
    output = tmp_path / filename
    materialize_experiment_matrix(
        f"configs/experiment/{filename}.yaml", output, config_root="configs"
    )
    repository = ConfigRepository("configs")
    recipes = {path.stem: repository.resolve(path) for path in output.glob("*.yaml")}
    assert set(recipes) == names
    reference = recipes["standard" if fraction == 0.1 else "seven-billion"]
    for recipe in recipes.values():
        assert recipe.training == reference.training
        assert recipe.data == reference.data
        assert recipe.runtime == reference.runtime
        assert recipe.model.adaptation == reference.model.adaptation
        assert recipe.data.sample_fraction == fraction
        assert recipe.initial_checkpoint_id is None
        assert recipe.model.vision.image_size == 336
        assert (
            recipe.training.per_device_batch_size
            * recipe.training.gradient_accumulation_steps
            * recipe.runtime.num_processes
        ) == 128
    assert recipes["hd"].model.language == reference.model.language
    assert recipes["hd"].model.vision.feature_dim == 2048
    assert recipes["hd"].model.vision.feature_layers == (-2, -1)
    if fraction == 0.1:
        assert reference.training.checkpoint_milestones == (0.1, 0.2, 0.5, 1.0)
        assert [round(520 * value) for value in reference.training.checkpoint_milestones] == [
            52,
            104,
            260,
            520,
        ]
        assert all("five-percent" not in recipe.tags for recipe in recipes.values())
    else:
        assert recipes["thirteen-billion"].model.language.hidden_size == 5120
        assert recipes["thirteen-billion"].model.language.num_layers == 40
        assert recipes["thirteen-billion"].model.vision == reference.model.vision


def test_ablation_matrix_materializes_atomically_with_evidence(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    materialize_experiment_matrix(
        "configs/experiment/ablation_matrix.yaml",
        output,
        config_root="configs",
    )

    manifest = json.loads((output / "matrix.manifest.json").read_text(encoding="utf-8"))
    source = yaml.safe_load((output / "operator-add.yaml").read_text(encoding="utf-8"))
    assert len(manifest["variants"]) == 18
    assert len({item["scientific_id"] for item in manifest["variants"]}) == 18
    assert source["description"].startswith("5% fusion-operator ablation")
    assert "factor:fusion-operator" in source["tags"]


def test_component_matrix_has_matched_depth_targets_and_random_projector(tmp_path: Path) -> None:
    output = tmp_path / "component-matrix"
    materialize_experiment_matrix(
        "configs/experiment/component_ablation_matrix.yaml", output, config_root="configs"
    )
    manifest = json.loads((output / "matrix.manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["variants"]) == 10
    assert len({item["scientific_id"] for item in manifest["variants"]}) == 10
    repository = ConfigRepository("configs")
    shallow = repository.resolve(output / "depth0-matched-lora.yaml")
    middle = repository.resolve(output / "depth16-matched-lora.yaml")
    assert shallow.model.adaptation == middle.model.adaptation
    assert shallow.model.adaptation.excluded_layer_indices == (0, 16)
    assert shallow.model.fusion.layers == (0,)
    assert middle.model.fusion.layers == (16,)
    conventional = repository.resolve(output / "single-stage-projector.yaml")
    assert conventional.model.projector.initialization == "random"
    assert conventional.model.projector.initial_checkpoint_id is None
    assert conventional.model.vision.feature_layers == (-1,)
    assert conventional.model.adaptation.excluded_layer_indices == (0,)


def test_feature_interface_matrix_has_four_matched_training_conditions(tmp_path: Path):
    output = tmp_path / "feature-interface"
    materialize_experiment_matrix(
        "configs/experiment/feature_interface_matrix.yaml", output, config_root="configs"
    )
    repository = ConfigRepository("configs")
    recipes = {path.stem: repository.resolve(path) for path in output.glob("*.yaml")}
    assert set(recipes) == {
        "inverse-final",
        "inverse-penultimate",
        "projector-final",
        "projector-penultimate",
    }
    reference = recipes["inverse-final"]
    for name, recipe in recipes.items():
        assert recipe.training == reference.training
        assert recipe.data == reference.data
        assert recipe.runtime == reference.runtime
        assert recipe.initial_checkpoint_id is None
        assert recipe.model.language == reference.model.language
        assert recipe.model.vision.feature_layers == ((-2,) if "penultimate" in name else (-1,))
        assert recipe.model.vision.feature_dim == 1024
        assert recipe.model.adaptation.rank == 128
        assert recipe.model.adaptation.trainable
        assert recipe.data.sample_fraction == 0.05
        assert recipe.data.sample_seed == 17
        assert (
            recipe.training.per_device_batch_size
            * recipe.training.gradient_accumulation_steps
            * recipe.runtime.num_processes
            == 128
        )
        if name.startswith("projector"):
            assert recipe.model.projector.initialization == "random"
            assert recipe.model.projector.initial_checkpoint_id is None
            assert recipe.model.adaptation.excluded_layer_indices == (0,)


def test_ocr_replay_matrix_keeps_canonical_parent_and_full_control_packages(tmp_path: Path):
    destination = tmp_path / "ocr-replay"
    materialize_experiment_matrix(
        "configs/experiment/ocr_replay_matrix.yaml", destination, config_root="configs"
    )
    repository = ConfigRepository("configs")
    base = repository.resolve("configs/experiment/paired_continuation_base.yaml")
    recipes = sorted(destination.glob("*.yaml"))
    assert len(recipes) == 3
    for path in recipes:
        recipe = repository.resolve(path)
        assert recipe.initial_checkpoint_id == base.initial_checkpoint_id
        assert recipe.model == base.model
        assert recipe.training == base.training
        assert recipe.runtime == base.runtime
        assert recipe.data.max_samples is None
        assert recipe.data.annotation.revision == "ocr-replay-controls-v1-expanded-targets"


def test_instruction_recovery_changes_only_parent_within_each_replay_dose(tmp_path: Path):
    output = tmp_path / "instruction-recovery"
    materialize_experiment_matrix(
        "configs/experiment/instruction_recovery_matrix.yaml", output, config_root="configs"
    )
    repository = ConfigRepository("configs")
    paired = repository.resolve("configs/experiment/paired_continuation_base.yaml")
    parents = {
        "parent": paired.initial_checkpoint_id,
        "paired1": "sha256:d77d91e08cb9e11aac30e24ab4626bcf91357bb87041eb6c3647dabebfa003e7",
        "paired5": "sha256:d0ddcd50c4cfd2b81050ccec62f720f34bd643bedab86a6b3763311484c6f2a3",
    }
    assert len(list(output.glob("*.yaml"))) == 6
    for fraction in (1, 10):
        control = repository.resolve(output / f"parent-replay-{fraction}pct.yaml")
        for history, parent_id in parents.items():
            recipe = repository.resolve(output / f"{history}-replay-{fraction}pct.yaml")
            assert recipe.initial_checkpoint_id == parent_id
            assert recipe.model == control.model == paired.model
            assert recipe.training == control.training == paired.training
            assert recipe.runtime == control.runtime == paired.runtime
            assert recipe.data == control.data
            assert recipe.data.sample_fraction == fraction / 100
            assert recipe.data.sample_seed == 17
            assert recipe.data.max_samples is None
            assert recipe.model.adaptation.trainable and recipe.model.fusion.trainable
            assert (
                recipe.training.per_device_batch_size
                * recipe.training.gradient_accumulation_steps
                * recipe.runtime.num_processes
            ) == 128
