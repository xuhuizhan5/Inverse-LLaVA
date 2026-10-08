import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from invllava.model.reference_audit import audit_llava_fft_conversion


def test_audit_llava_fft_conversion_checks_exact_tensors(tmp_path: Path) -> None:
    official = tmp_path / "official"
    converted = tmp_path / "converted"
    official.mkdir()
    converted.mkdir()
    original = {
        "model.embed_tokens.weight": torch.randn(4, 3),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(3, 3),
        "model.layers.0.self_attn.rotary_emb.inv_freq": torch.randn(2),
        "model.mm_projector.0.weight": torch.randn(3, 2),
        "model.mm_projector.0.bias": torch.randn(3),
        "model.mm_projector.2.weight": torch.randn(3, 3),
        "model.mm_projector.2.bias": torch.randn(3),
        "model.norm.weight": torch.randn(3),
        "lm_head.weight": torch.randn(4, 3),
    }
    torch.save(original, official / "pytorch_model-00001-of-00001.bin")
    (official / "pytorch_model.bin.index.json").write_text(
        json.dumps({"weight_map": {name: "pytorch_model-00001-of-00001.bin" for name in original}}),
        encoding="utf-8",
    )
    torch.save(
        {name: value for name, value in original.items() if "mm_projector" in name},
        official / "mm_projector.bin",
    )
    mapped = {
        "language_model.model.embed_tokens.weight": torch.cat(
            (original["model.embed_tokens.weight"], torch.randn(2, 3))
        ),
        "language_model.model.layers.0.self_attn.q_proj.weight": original[
            "model.layers.0.self_attn.q_proj.weight"
        ],
        "language_model.model.norm.weight": original["model.norm.weight"],
        "language_model.lm_head.weight": torch.cat((original["lm_head.weight"], torch.randn(2, 3))),
        "multi_modal_projector.linear_1.weight": original["model.mm_projector.0.weight"],
        "multi_modal_projector.linear_1.bias": original["model.mm_projector.0.bias"],
        "multi_modal_projector.linear_2.weight": original["model.mm_projector.2.weight"],
        "multi_modal_projector.linear_2.bias": original["model.mm_projector.2.bias"],
        "vision_tower.fixture": torch.randn(1),
    }
    save_file(mapped, converted / "model-00001-of-00001.safetensors")
    (converted / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model-00001-of-00001.safetensors" for name in mapped}}),
        encoding="utf-8",
    )

    report = audit_llava_fft_conversion(official, converted, output=tmp_path / "report.json")
    assert report["passed"] is True
    assert report["compared_tensors"] == 8
    assert report["ignored_derived_rotary_buffers"] == 1
    assert report["standalone_projector_tensors"] == 4
    assert set(report["padded_vocab_rows"].values()) == {2}
    inventory = report["converted_inventory"]
    assert inventory["elements"] == 70
    assert inventory["components"] == {
        "language_model": {"tensors": 4, "elements": 48},
        "multi_modal_projector": {"tensors": 4, "elements": 21},
        "vision_tower": {"tensors": 1, "elements": 1},
    }
