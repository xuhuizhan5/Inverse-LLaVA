import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from invllava.model.champion_checkpoint import (
    RELEASE_CONFIG_FILENAME,
    convert_champion_checkpoint,
)


def _model_config() -> dict[str, object]:
    return {
        "id": "champion-fixture",
        "architecture": "inverse_llava",
        "language": {
            "checkpoint": "fixture/base",
            "revision": "a" * 40,
            "hidden_size": 8,
            "num_layers": 2,
            "max_length": 32,
        },
        "vision": {
            "checkpoint": "fixture/vision",
            "revision": "b" * 40,
            "feature_layers": [-1],
            "feature_dim": 2,
            "image_size": 8,
            "aspect_ratio": "pad",
        },
        "fusion": {
            "layers": [0],
            "targets": ["q", "k", "v"],
            "operator": "concat",
            "mapper_rank": None,
            "mapper_trainable": True,
            "visual_normalization": "rms",
            "visual_norm_eps": 1e-6,
            "scale_mode": "learned",
            "initial_scale": 1.0,
            "initialization_id": "fixture",
        },
        "adaptation": {
            "method": "lora",
            "rank": 2,
            "alpha": 4,
            "dropout": 0.05,
            "target_suffixes": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "exclude_fusion_layers": True,
        },
        "torch_dtype": "bfloat16",
    }


def _write_fixture(root: Path) -> Path:
    source = root / "author-checkpoint"
    source.mkdir()
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    adapter = {}
    for module in target_modules:
        family = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
        prefix = f"base_model.model.model.layers.1.{family}.{module}"
        adapter[f"{prefix}.lora_A.weight"] = torch.randn(2, 8)
        adapter[f"{prefix}.lora_B.weight"] = torch.randn(8, 2)
    save_file(adapter, source / "adapter_model.safetensors")
    (source / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "bias": "none",
                "fan_in_fan_out": False,
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.05,
                "base_model_name_or_path": "fixture/base",
                "target_modules": target_modules,
                "layers_to_transform": [1],
            }
        ),
        encoding="utf-8",
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fusion_llama",
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "mm_hidden_size": 2,
                "mm_vision_tower": "fixture/vision",
                "mm_vision_select_feature": "patch",
                "mm_vision_select_layer": -1,
                "image_aspect_ratio": "pad",
                "use_vision_fusion": True,
            }
        ),
        encoding="utf-8",
    )
    complete = {}
    for target in ("q", "k", "v"):
        prefix = f"base_model.model.model.layers.0.self_attn.{target}_proj"
        complete[f"{prefix}.original_weight"] = torch.randn(8, 8)
        complete[f"{prefix}.up_A"] = torch.randn(8, 2)
        complete[f"{prefix}.down_B"] = torch.randn(4, 8)
        complete[f"{prefix}.alpha"] = torch.tensor(1.0)
        complete[f"{prefix}.vision_norm.weight"] = torch.ones(2)
    complete["base_model.model.model.layers.0.self_attn.o_proj.weight"] = torch.randn(8, 8)
    compact = {
        name: tensor
        for name, tensor in complete.items()
        if not name.endswith("original_weight") and not name.endswith("o_proj.weight")
    }
    torch.save(compact, source / "non_lora_trainables.bin")
    torch.save(complete, source / "non_lora_trainables_skip_layer.bin")
    (source / "trainer_state.json").write_text(
        json.dumps(
            {
                "epoch": 1.0,
                "global_step": 10,
                "max_steps": 10,
                "train_batch_size": 2,
                "log_history": [{"train_loss": 0.5, "train_runtime": 3.0}],
            }
        ),
        encoding="utf-8",
    )
    return source


def test_convert_champion_writes_one_safe_native_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_fixture(tmp_path)
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump(_model_config()), encoding="utf-8")
    model_card = tmp_path / "README.md"
    model_card.write_text("# Fixture\n", encoding="utf-8")
    complete = torch.load(
        source / "non_lora_trainables_skip_layer.bin",
        map_location="cpu",
        weights_only=True,
    )
    upstream = {
        "language_model.model.layers.0.self_attn.q_proj.weight": complete[
            "base_model.model.model.layers.0.self_attn.q_proj.original_weight"
        ],
        "language_model.model.layers.0.self_attn.k_proj.weight": complete[
            "base_model.model.model.layers.0.self_attn.k_proj.original_weight"
        ],
        "language_model.model.layers.0.self_attn.v_proj.weight": complete[
            "base_model.model.model.layers.0.self_attn.v_proj.original_weight"
        ],
        "language_model.model.layers.0.self_attn.o_proj.weight": complete[
            "base_model.model.model.layers.0.self_attn.o_proj.weight"
        ],
    }
    monkeypatch.setattr(
        "invllava.model.champion_checkpoint._load_upstream_projection_state",
        lambda model: upstream,
    )
    output = convert_champion_checkpoint(
        source=source,
        model_config=model_config,
        destination=tmp_path / "release",
        model_card=model_card,
    )

    assert (output / "COMPLETE").read_text(encoding="utf-8") == "complete\n"
    assert output.stat().st_mode & 0o777 == 0o755
    assert (output / "README.md").read_text(encoding="utf-8") == "# Fixture\n"
    assert (
        json.loads((output / RELEASE_CONFIG_FILENAME).read_text(encoding="utf-8"))["format"]
        == "invllava-hub-release-v1"
    )
    tensors = load_file(output / "model_delta.safetensors")
    assert len(tensors) == 26
    assert tensors[
        "language_model.model.layers.0.self_attn.fusion.text_to_vision.q.weight"
    ].shape == (2, 8)
    assert tensors["language_model.model.layers.0.self_attn.fusion.output.q.weight"].shape == (8, 4)
    assert "language_model.model.layers.0.self_attn.q_proj.weight" not in tensors
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tensor_count"] == 26
    assert metadata["training_summary"]["global_step"] == 10
    assert metadata["upstream_base_projection_verification"]["status"] == "exact"
    checksum_names = {
        line.split("  ", 1)[1]
        for line in (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    }
    assert checksum_names == {
        "COMPLETE",
        "README.md",
        RELEASE_CONFIG_FILENAME,
        "metadata.json",
        "model_delta.safetensors",
    }
