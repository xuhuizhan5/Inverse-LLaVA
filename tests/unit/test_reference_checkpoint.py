import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from invllava.model.reference_checkpoint import convert_official_llava_lora


def _write_reference_files(root: Path, *, unexpected_non_lora: bool = False) -> None:
    adapter = {
        "base_model.model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(
            2, 4
        ),
        "base_model.model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(
            4, 2
        ),
    }
    non_lora = {
        "base_model.model.model.mm_projector.0.weight": torch.randn(4, 3),
        "base_model.model.model.mm_projector.0.bias": torch.randn(4),
        "base_model.model.model.mm_projector.2.weight": torch.randn(4, 4),
        "base_model.model.model.mm_projector.2.bias": torch.randn(4),
    }
    if unexpected_non_lora:
        non_lora["base_model.model.model.embed_tokens.weight"] = torch.randn(4, 4)
    torch.save(adapter, root / "adapter_model.bin")
    torch.save(non_lora, root / "non_lora_trainables.bin")
    (root / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "bias": "none",
                "fan_in_fan_out": False,
                "r": 2,
                "lora_alpha": 4,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj"],
                "base_model_name_or_path": "fixture/base",
            }
        ),
        encoding="utf-8",
    )


def test_convert_official_llava_lora_maps_adapter_and_projector(tmp_path: Path) -> None:
    _write_reference_files(tmp_path)
    output = convert_official_llava_lora(
        adapter_model=tmp_path / "adapter_model.bin",
        adapter_config=tmp_path / "adapter_config.json",
        non_lora_trainables=tmp_path / "non_lora_trainables.bin",
        destination=tmp_path / "converted",
    )

    assert (output / "COMPLETE").read_text(encoding="utf-8") == "complete\n"
    tensors = load_file(output / "model_delta.safetensors")
    assert set(tensors) == {
        "language_model.model.layers.0.self_attn.q_proj.lora_a.weight",
        "language_model.model.layers.0.self_attn.q_proj.lora_b.weight",
        "multimodal_projector.linear_1.weight",
        "multimodal_projector.linear_1.bias",
        "multimodal_projector.linear_2.weight",
        "multimodal_projector.linear_2.bias",
    }
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rank"] == 2
    assert metadata["alpha"] == 4
    assert metadata["tensor_count"] == 6


def test_convert_official_llava_lora_rejects_unmapped_trainables(tmp_path: Path) -> None:
    _write_reference_files(tmp_path, unexpected_non_lora=True)
    with pytest.raises(ValueError, match="unsupported non-LoRA tensors"):
        convert_official_llava_lora(
            adapter_model=tmp_path / "adapter_model.bin",
            adapter_config=tmp_path / "adapter_config.json",
            non_lora_trainables=tmp_path / "non_lora_trainables.bin",
            destination=tmp_path / "converted",
        )
