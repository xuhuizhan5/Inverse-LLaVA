"""Safe, explicit conversion of LLaVA projector-only checkpoints."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file

PROJECTOR_FILENAME = "projector.safetensors"

_ALIASES = {
    "mm_projector.0.weight": "linear_1.weight",
    "mm_projector.0.bias": "linear_1.bias",
    "mm_projector.2.weight": "linear_2.weight",
    "mm_projector.2.bias": "linear_2.bias",
    "multi_modal_projector.linear_1.weight": "linear_1.weight",
    "multi_modal_projector.linear_1.bias": "linear_1.bias",
    "multi_modal_projector.linear_2.weight": "linear_2.weight",
    "multi_modal_projector.linear_2.bias": "linear_2.bias",
}
_EXPECTED = frozenset(_ALIASES.values())


def _load_source(path: Path) -> dict[str, Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        value: Any = load_file(path)
    else:
        value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict) and isinstance(value.get("state_dict"), dict):
        value = value["state_dict"]
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    ):
        raise ValueError("projector source must be a flat tensor state dictionary")
    return value


def extract_projector_state(source: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map original-LLaVA and Hugging Face projector names to one contract."""

    extracted: dict[str, Tensor] = {}
    origins: dict[str, str] = {}
    for source_name, tensor in source.items():
        matches = [target for suffix, target in _ALIASES.items() if source_name.endswith(suffix)]
        if not matches:
            continue
        target = matches[0]
        if target in extracted:
            raise ValueError(
                f"multiple projector tensors map to {target}: {origins[target]} and {source_name}"
            )
        extracted[target] = tensor.detach().cpu().contiguous()
        origins[target] = source_name
    missing = _EXPECTED - extracted.keys()
    if missing:
        raise ValueError(f"projector source is missing tensors: {sorted(missing)}")
    if extracted.keys() != _EXPECTED:
        raise ValueError("projector extraction produced an unexpected tensor set")
    first = extracted["linear_1.weight"]
    second = extracted["linear_2.weight"]
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("projector weights must be matrices")
    if tuple(extracted["linear_1.bias"].shape) != (first.shape[0],):
        raise ValueError("first projector bias shape is invalid")
    if tuple(second.shape) != (first.shape[0], first.shape[0]):
        raise ValueError("second projector weight must map within the language hidden size")
    if tuple(extracted["linear_2.bias"].shape) != (second.shape[0],):
        raise ValueError("second projector bias shape is invalid")
    return extracted


def convert_projector_checkpoint(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent)
    )
    try:
        from safetensors.torch import save_file

        state = extract_projector_state(_load_source(source_path))
        save_file(state, temporary / PROJECTOR_FILENAME)
        atomic_write_json(
            temporary / "metadata.json",
            {
                "format": "invllava-projector-v1",
                "source_name": source_path.name,
                "source_sha256": sha256_file(source_path),
                "input_dim": state["linear_1.weight"].shape[1],
                "output_dim": state["linear_1.weight"].shape[0],
                "tensor_names": sorted(state),
            },
        )
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        os.replace(temporary, destination_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination_path


def load_projector_weights(path: str | Path, projector: nn.Module) -> None:
    source = Path(path)
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"incomplete projector checkpoint: {source}")
    from safetensors.torch import load_file

    state = load_file(source / PROJECTOR_FILENAME)
    projector.load_state_dict(state, strict=True)
