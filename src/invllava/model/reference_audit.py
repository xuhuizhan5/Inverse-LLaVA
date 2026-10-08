"""Byte-equivalence audits for official and interoperable LLaVA checkpoints."""

from __future__ import annotations

import json
from contextlib import ExitStack
from math import prod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file


def _mapped_name(name: str) -> str | None:
    if name.endswith(".rotary_emb.inv_freq"):
        return None
    projector = {
        "model.mm_projector.0.weight": "multi_modal_projector.linear_1.weight",
        "model.mm_projector.0.bias": "multi_modal_projector.linear_1.bias",
        "model.mm_projector.2.weight": "multi_modal_projector.linear_2.weight",
        "model.mm_projector.2.bias": "multi_modal_projector.linear_2.bias",
    }
    if name in projector:
        return projector[name]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    if name.startswith("model."):
        return "language_model." + name
    raise ValueError(f"unsupported official FFT tensor: {name}")


def _load_index(path: Path, filename: str) -> dict[str, str]:
    value = json.loads((path / filename).read_text(encoding="utf-8"))
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in weight_map.items()
    ):
        raise ValueError(f"invalid checkpoint index: {path / filename}")
    return weight_map


def _tensor_dict(path: Path) -> dict[str, Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(tensor, Tensor) for key, tensor in value.items()
    ):
        raise ValueError(f"expected a flat tensor state dictionary: {path}")
    return value


def _checkpoint_component(name: str) -> str:
    for component in ("language_model", "multi_modal_projector", "vision_tower"):
        if name == component or name.startswith(component + "."):
            return component
    return "other"


def safetensors_index_inventory(root: str | Path) -> dict[str, Any]:
    """Count tensors and elements from safetensors metadata without loading weights."""

    checkpoint_root = Path(root).resolve()
    weight_map = _load_index(checkpoint_root, "model.safetensors.index.json")
    from safetensors import safe_open

    by_component: dict[str, dict[str, int]] = {}
    dtypes: dict[str, int] = {}
    total_elements = 0
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(checkpoint_root / filename, framework="pt", device="cpu")
            )
            for filename in set(weight_map.values())
        }
        for name, filename in weight_map.items():
            reader = readers[filename]
            if name not in reader.keys():
                raise ValueError(f"checkpoint index points to a missing tensor: {name}")
            tensor_slice = reader.get_slice(name)
            elements = prod(tensor_slice.get_shape())
            component = _checkpoint_component(name)
            counts = by_component.setdefault(component, {"tensors": 0, "elements": 0})
            counts["tensors"] += 1
            counts["elements"] += elements
            dtype = str(tensor_slice.get_dtype())
            dtypes[dtype] = dtypes.get(dtype, 0) + elements
            total_elements += elements
    return {
        "index_sha256": sha256_file(checkpoint_root / "model.safetensors.index.json"),
        "tensors": len(weight_map),
        "elements": total_elements,
        "elements_by_dtype": dict(sorted(dtypes.items())),
        "components": dict(sorted(by_component.items())),
    }


def audit_llava_fft_conversion(
    official: str | Path,
    converted: str | Path,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Prove exact language/projector tensor equality across checkpoint formats."""

    original_root = Path(official).resolve()
    converted_root = Path(converted).resolve()
    original_index = _load_index(original_root, "pytorch_model.bin.index.json")
    converted_index = _load_index(converted_root, "model.safetensors.index.json")
    mapped = {name: target for name in original_index if (target := _mapped_name(name))}
    expected = set(mapped.values())
    observed = {
        name
        for name in converted_index
        if name.startswith("language_model.") or name.startswith("multi_modal_projector.")
    }
    if expected != observed:
        raise ValueError(
            "converted checkpoint language/projector key mismatch; "
            f"missing={sorted(expected - observed)[:8]}, extra={sorted(observed - expected)[:8]}"
        )

    from safetensors import safe_open

    compared = 0
    elements = 0
    dtypes: set[str] = set()
    padded_vocab_rows: dict[str, int] = {}
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(converted_root / filename, framework="pt", device="cpu")
            )
            for filename in set(converted_index.values())
        }
        for filename in sorted(set(original_index.values())):
            shard = _tensor_dict(original_root / filename)
            expected_names = {name for name, source in original_index.items() if source == filename}
            if set(shard) != expected_names:
                raise ValueError(f"official shard does not match its index: {filename}")
            for name, tensor in shard.items():
                target = _mapped_name(name)
                if target is None:
                    continue
                converted_tensor = readers[converted_index[target]].get_tensor(target)
                permits_vocab_padding = name in {"model.embed_tokens.weight", "lm_head.weight"}
                if (
                    permits_vocab_padding
                    and tensor.ndim == 2
                    and converted_tensor.ndim == 2
                    and converted_tensor.shape[0] >= tensor.shape[0]
                    and converted_tensor.shape[1] == tensor.shape[1]
                    and tensor.dtype == converted_tensor.dtype
                ):
                    comparable = converted_tensor[: tensor.shape[0]]
                    padded_vocab_rows[target] = converted_tensor.shape[0] - tensor.shape[0]
                else:
                    comparable = converted_tensor
                if tensor.shape != comparable.shape or tensor.dtype != comparable.dtype:
                    raise ValueError(f"tensor metadata differs for {name} -> {target}")
                if not torch.equal(tensor, comparable):
                    raise ValueError(f"tensor values differ for {name} -> {target}")
                compared += 1
                elements += tensor.numel()
                dtypes.add(str(tensor.dtype))
            del shard

    projector = _tensor_dict(original_root / "mm_projector.bin")
    projector_aliases = {
        "model.mm_projector.0.weight": "multi_modal_projector.linear_1.weight",
        "model.mm_projector.0.bias": "multi_modal_projector.linear_1.bias",
        "model.mm_projector.2.weight": "multi_modal_projector.linear_2.weight",
        "model.mm_projector.2.bias": "multi_modal_projector.linear_2.bias",
    }
    if set(projector) != set(projector_aliases):
        raise ValueError("official standalone projector contains an unexpected tensor set")
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(converted_root / filename, framework="pt", device="cpu")
            )
            for filename in set(converted_index.values())
        }
        for name, tensor in projector.items():
            target = projector_aliases[name]
            if not torch.equal(tensor, readers[converted_index[target]].get_tensor(target)):
                raise ValueError(f"standalone projector differs for {name}")

    report = {
        "schema_version": 1,
        "passed": True,
        "official_root": str(original_root),
        "converted_root": str(converted_root),
        "official_index_sha256": sha256_file(original_root / "pytorch_model.bin.index.json"),
        "converted_index_sha256": sha256_file(converted_root / "model.safetensors.index.json"),
        "compared_tensors": compared,
        "compared_elements": elements,
        "ignored_derived_rotary_buffers": len(original_index) - compared,
        "dtypes": sorted(dtypes),
        "padded_vocab_rows": padded_vocab_rows,
        "standalone_projector_tensors": len(projector),
        "embedded_vision_tensors": sum(
            name.startswith("vision_tower.") for name in converted_index
        ),
        "converted_inventory": safetensors_index_inventory(converted_root),
    }
    if output is not None:
        atomic_write_json(output, report)
    return report
