"""Conversion of the official LLaVA-1.5 LoRA release into the native delta contract."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.model.projector_checkpoint import extract_projector_state

REFERENCE_PROJECTOR_FILENAME = "projector.safetensors"
REFERENCE_DELTA_FILENAME = "model_delta.safetensors"
_LORA_KEY = re.compile(
    r"(?:^|\.)(model\.layers\.[0-9]+\.(?:self_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\."
    r"lora_([AB])(?:\.default)?\.weight$"
)
_PROJECTOR_SUFFIXES = (
    "mm_projector.0.weight",
    "mm_projector.0.bias",
    "mm_projector.2.weight",
    "mm_projector.2.bias",
    "multi_modal_projector.linear_1.weight",
    "multi_modal_projector.linear_1.bias",
    "multi_modal_projector.linear_2.weight",
    "multi_modal_projector.linear_2.bias",
)


def _load_tensor_dict(path: Path) -> dict[str, Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    ):
        raise ValueError(f"expected a flat tensor state dictionary: {path}")
    return value


def normalize_official_lora_state(source: dict[str, Tensor]) -> dict[str, Tensor]:
    normalized: dict[str, Tensor] = {}
    for source_name, tensor in source.items():
        match = _LORA_KEY.search(source_name)
        if match is None:
            raise ValueError(f"unsupported official LoRA tensor: {source_name}")
        module_name, branch = match.groups()
        target = f"language_model.{module_name}.lora_{branch.lower()}.weight"
        if target in normalized:
            raise ValueError(f"multiple official LoRA tensors map to {target}")
        normalized[target] = tensor.detach().cpu().contiguous()
    if not normalized:
        raise ValueError("official LoRA checkpoint contains no adapter tensors")
    return normalized


def convert_official_llava_lora(
    *,
    adapter_model: str | Path,
    adapter_config: str | Path,
    non_lora_trainables: str | Path,
    destination: str | Path,
) -> Path:
    adapter_path = Path(adapter_model).resolve()
    config_path = Path(adapter_config).resolve()
    non_lora_path = Path(non_lora_trainables).resolve()
    target = Path(destination).resolve()
    for path in (adapter_path, config_path, non_lora_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if target.exists():
        raise FileExistsError(target)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "fan_in_fan_out": False,
    }
    mismatches = {
        key: (config.get(key), value) for key, value in required.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"unsupported official adapter configuration: {mismatches}")
    rank = int(config.get("r", 0))
    alpha = int(config.get("lora_alpha", 0))
    if rank <= 0 or alpha <= 0:
        raise ValueError("official adapter rank and alpha must be positive")

    adapter = normalize_official_lora_state(_load_tensor_dict(adapter_path))
    non_lora = _load_tensor_dict(non_lora_path)
    unsupported = sorted(
        name
        for name in non_lora
        if not any(name.endswith(suffix) for suffix in _PROJECTOR_SUFFIXES)
    )
    if unsupported:
        raise ValueError(f"unsupported non-LoRA tensors: {unsupported[:8]}")
    projector = extract_projector_state(non_lora)
    combined = {
        **adapter,
        **{f"multimodal_projector.{name}": tensor for name, tensor in projector.items()},
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        from safetensors.torch import save_file

        save_file(combined, temporary / REFERENCE_DELTA_FILENAME)
        save_file(projector, temporary / REFERENCE_PROJECTOR_FILENAME)
        atomic_write_json(
            temporary / "metadata.json",
            {
                "format": "invllava-official-llava-lora-reference-v1",
                "adapter_model_sha256": sha256_file(adapter_path),
                "adapter_config_sha256": sha256_file(config_path),
                "non_lora_trainables_sha256": sha256_file(non_lora_path),
                "projector_sha256": sha256_file(temporary / REFERENCE_PROJECTOR_FILENAME),
                "model_delta_sha256": sha256_file(temporary / REFERENCE_DELTA_FILENAME),
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "rank": rank,
                "alpha": alpha,
                "dropout": float(config.get("lora_dropout", 0.0)),
                "target_modules": sorted(str(value) for value in config.get("target_modules", [])),
                "tensor_count": len(combined),
            },
        )
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
