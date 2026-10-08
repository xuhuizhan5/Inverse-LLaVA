"""Strict conversion of the paper checkpoint into the public release format.

The checkpoint combines a PEFT safetensors adapter with two PyTorch tensor
dictionaries. Training, inference, and Hub loading consume the converted
``model_delta.safetensors`` representation.
"""

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
from invllava.config.schema import ModelSpec
from invllava.model.reference_checkpoint import normalize_official_lora_state
from invllava.train.checkpoint import MODEL_DELTA_FILENAME

RELEASE_CONFIG_FILENAME = "inverse_llava_config.json"
RELEASE_FORMAT = "invllava-hub-release-v1"
CHAMPION_SOURCE_FORMAT = "inverse-llava-author-peft-fusion-v1"

_FUSION_KEY = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>[0-9]+)\.self_attn\."
    r"(?P<target>q|k|v)_proj\."
    r"(?P<component>up_A|down_B|alpha|vision_norm\.weight|original_weight)$"
)
_OUTPUT_KEY = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>[0-9]+)\.self_attn\.o_proj\.weight$"
)


def _load_tensor_dict(path: Path) -> dict[str, Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, Tensor) for name, tensor in value.items()
    ):
        raise ValueError(f"expected a flat tensor state dictionary: {path}")
    return value


def _required_source_files(root: Path) -> dict[str, Path]:
    names = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "config.json",
        "non_lora_trainables.bin",
        "non_lora_trainables_skip_layer.bin",
        "trainer_state.json",
    )
    files = {name: root / name for name in names}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"champion checkpoint is missing files: {missing}")
    return files


def _validate_adapter_config(config: dict[str, Any], model: ModelSpec) -> None:
    expected_targets = set(model.adaptation.target_suffixes)
    observed_targets = {str(value) for value in config.get("target_modules", ())}
    expected_layers = set(range(model.language.num_layers)) - set(model.fusion.layers)
    observed_layers = {int(value) for value in config.get("layers_to_transform", ())}
    mismatches: dict[str, object] = {}
    checks = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "fan_in_fan_out": False,
        "r": model.adaptation.rank,
        "lora_alpha": model.adaptation.alpha,
        "lora_dropout": model.adaptation.dropout,
        "base_model_name_or_path": model.language.checkpoint,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            mismatches[key] = {"observed": config.get(key), "expected": expected}
    if observed_targets != expected_targets:
        mismatches["target_modules"] = {
            "observed": sorted(observed_targets),
            "expected": sorted(expected_targets),
        }
    if observed_layers != expected_layers:
        mismatches["layers_to_transform"] = {
            "observed": sorted(observed_layers),
            "expected": sorted(expected_layers),
        }
    if mismatches:
        raise ValueError(f"champion adapter disagrees with model configuration: {mismatches}")


def _validate_model_config(config: dict[str, Any], model: ModelSpec) -> None:
    checks = {
        "model_type": "fusion_llama",
        "hidden_size": model.language.hidden_size,
        "num_hidden_layers": model.language.num_layers,
        "mm_hidden_size": model.vision.feature_dim,
        "mm_vision_tower": model.vision.checkpoint,
        "mm_vision_select_feature": model.vision.feature_select,
        "image_aspect_ratio": model.vision.aspect_ratio,
    }
    mismatches = {
        key: {"observed": config.get(key), "expected": expected}
        for key, expected in checks.items()
        if config.get(key) != expected
    }
    feature_layers = tuple(model.vision.feature_layers)
    if len(feature_layers) != 1 or config.get("mm_vision_select_layer") != feature_layers[0]:
        mismatches["mm_vision_select_layer"] = {
            "observed": config.get("mm_vision_select_layer"),
            "expected": list(feature_layers),
        }
    if not config.get("use_vision_fusion"):
        mismatches["use_vision_fusion"] = {
            "observed": config.get("use_vision_fusion"),
            "expected": True,
        }
    if mismatches:
        raise ValueError(f"champion architecture disagrees with model configuration: {mismatches}")


def _validate_redundant_state(compact: dict[str, Tensor], complete: dict[str, Tensor]) -> None:
    if set(compact) - set(complete):
        raise ValueError("compact non-LoRA state contains tensors absent from the complete state")
    unequal = [name for name in compact if not torch.equal(compact[name], complete[name])]
    if unequal:
        raise ValueError(f"duplicated non-LoRA tensors disagree: {unequal[:8]}")


def _validate_adapter_state(adapter: dict[str, Tensor], model: ModelSpec) -> None:
    expected: set[str] = set()
    for layer in set(range(model.language.num_layers)) - set(model.fusion.layers):
        for module in model.adaptation.target_suffixes:
            family = "self_attn" if module in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"language_model.model.layers.{layer}.{family}.{module}"
            expected.update({f"{prefix}.lora_a.weight", f"{prefix}.lora_b.weight"})
    if set(adapter) != expected:
        raise ValueError(
            "champion adapter tensor inventory disagrees with the configured LoRA policy; "
            f"missing={sorted(expected - set(adapter))[:8]}, "
            f"unexpected={sorted(set(adapter) - expected)[:8]}"
        )
    rank = model.adaptation.rank
    malformed = []
    for name, tensor in adapter.items():
        if tensor.ndim != 2:
            malformed.append(name)
        elif name.endswith(".lora_a.weight") and tensor.shape[0] != rank:
            malformed.append(name)
        elif name.endswith(".lora_b.weight") and tensor.shape[1] != rank:
            malformed.append(name)
    if malformed:
        raise ValueError(f"champion adapter tensors have invalid rank or shape: {malformed[:8]}")


def normalize_champion_fusion_state(
    source: dict[str, Tensor],
    *,
    model: ModelSpec,
) -> dict[str, Tensor]:
    """Map the target-specific source fusion tensors to native names."""

    if model.fusion.mapper_rank is not None:
        raise ValueError("the champion artifact contains full text-to-vision matrices")
    if model.fusion.operator != "concat":
        raise ValueError("the champion artifact requires concat fusion")
    if model.fusion.visual_normalization != "rms":
        raise ValueError("the champion artifact requires per-target visual RMS normalization")
    expected_layers = set(model.fusion.layers)
    expected_targets = set(model.fusion.targets)
    observed: dict[tuple[int, str], set[str]] = {}
    normalized: dict[str, Tensor] = {}
    output_layers: set[int] = set()
    for source_name, source_tensor in source.items():
        tensor = source_tensor.detach().cpu().contiguous()
        match = _FUSION_KEY.fullmatch(source_name)
        if match is not None:
            layer = int(match.group("layer"))
            target = match.group("target")
            component = match.group("component")
            if layer not in expected_layers or target not in expected_targets:
                raise ValueError(f"unexpected champion fusion tensor: {source_name}")
            observed.setdefault((layer, target), set()).add(component)
            prefix = f"language_model.model.layers.{layer}.self_attn"
            if component == "up_A":
                expected_shape = (model.language.hidden_size, model.vision.feature_dim)
                destination = f"{prefix}.fusion.text_to_vision.{target}.weight"
                tensor = tensor.transpose(0, 1).contiguous()
            elif component == "down_B":
                expected_shape = (2 * model.vision.feature_dim, model.language.hidden_size)
                destination = f"{prefix}.fusion.output.{target}.weight"
                tensor = tensor.transpose(0, 1).contiguous()
            elif component == "alpha":
                expected_shape = ()
                destination = f"{prefix}.fusion.scales.{target}"
            elif component == "vision_norm.weight":
                expected_shape = (model.vision.feature_dim,)
                destination = f"{prefix}.fusion.visual_norm.{target}.weight"
            else:
                expected_shape = (model.language.hidden_size, model.language.hidden_size)
                destination = f"{prefix}.{target}_proj.weight"
            if tuple(source_tensor.shape) != expected_shape:
                raise ValueError(
                    f"champion tensor {source_name} has shape {tuple(source_tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            if destination in normalized:
                raise ValueError(f"multiple champion tensors map to {destination}")
            normalized[destination] = tensor
            continue
        output = _OUTPUT_KEY.fullmatch(source_name)
        if output is not None:
            layer = int(output.group("layer"))
            if layer not in expected_layers:
                raise ValueError(f"unexpected champion output tensor: {source_name}")
            expected_shape = (model.language.hidden_size, model.language.hidden_size)
            if tuple(source_tensor.shape) != expected_shape:
                raise ValueError(
                    f"champion tensor {source_name} has shape {tuple(source_tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            output_layers.add(layer)
            normalized[f"language_model.model.layers.{layer}.self_attn.o_proj.weight"] = tensor
            continue
        raise ValueError(f"unsupported champion non-LoRA tensor: {source_name}")

    required_components = {
        "up_A",
        "down_B",
        "alpha",
        "vision_norm.weight",
        "original_weight",
    }
    expected_branches = {
        (layer, target) for layer in expected_layers for target in expected_targets
    }
    if set(observed) != expected_branches:
        raise ValueError(
            f"champion fusion branches disagree; observed={sorted(observed)}, "
            f"expected={sorted(expected_branches)}"
        )
    incomplete = {
        branch: sorted(required_components - observed[branch])
        for branch in expected_branches
        if observed[branch] != required_components
    }
    if incomplete:
        raise ValueError(f"champion fusion branches are incomplete: {incomplete}")
    if output_layers != expected_layers:
        raise ValueError(
            "champion checkpoint does not preserve each fusion-layer output projection"
        )
    return normalized


def _training_summary(trainer_state: dict[str, Any]) -> dict[str, Any]:
    tail = trainer_state.get("log_history", ())
    final = tail[-1] if isinstance(tail, list) and tail else {}
    return {
        "epoch": trainer_state.get("epoch"),
        "global_step": trainer_state.get("global_step"),
        "max_steps": trainer_state.get("max_steps"),
        "trainer_reported_train_batch_size": trainer_state.get("train_batch_size"),
        "train_loss": final.get("train_loss"),
        "train_runtime_seconds": final.get("train_runtime"),
        "train_samples_per_second": final.get("train_samples_per_second"),
        "train_steps_per_second": final.get("train_steps_per_second"),
        "total_flos": final.get("total_flos"),
    }


def _partition_normalized_champion_state(
    state: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Separate learned deltas from copied layer-0 base projections."""

    learned = {name: tensor for name, tensor in state.items() if ".fusion." in name}
    copied_base = {name: tensor for name, tensor in state.items() if ".fusion." not in name}
    if not learned or not copied_base or len(copied_base) != 4:
        raise ValueError("champion non-LoRA state has an unexpected learned/base partition")
    return learned, copied_base


def _load_upstream_projection_state(model: ModelSpec) -> dict[str, Tensor]:
    """Resolve the four pinned base projections copied into the source artifact."""

    from invllava.model.loaders import build_language_model

    language, _ = build_language_model(model, attention_backend="eager")
    layer = language.model.layers[model.fusion.layers[0]].self_attn
    state = {
        f"language_model.model.layers.{model.fusion.layers[0]}.self_attn.{name}.weight": (
            getattr(layer, name).weight.detach().cpu().contiguous()
        )
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    return state


def _verify_upstream_projections(
    copied_base: dict[str, Tensor], model: ModelSpec
) -> dict[str, Any]:
    if len(model.fusion.layers) != 1:
        raise ValueError("champion upstream verification expects exactly one fusion layer")
    upstream = _load_upstream_projection_state(model)
    if set(upstream) != set(copied_base):
        raise ValueError(
            "champion and upstream projection inventories disagree; "
            f"missing={sorted(set(copied_base) - set(upstream))}, "
            f"unexpected={sorted(set(upstream) - set(copied_base))}"
        )
    unequal = [name for name in copied_base if not torch.equal(copied_base[name], upstream[name])]
    if unequal:
        raise ValueError(
            f"champion copied base projections differ from the pinned Vicuna revision: {unequal}"
        )
    return {
        "status": "exact",
        "base_model": model.language.checkpoint,
        "base_model_revision": model.language.revision,
        "tensor_names": sorted(copied_base),
        "omitted_from_release": True,
    }


def convert_champion_checkpoint(
    *,
    source: str | Path,
    model_config: str | Path,
    destination: str | Path,
    model_card: str | Path | None = None,
) -> Path:
    """Convert and seal the author checkpoint as one safe Hub-ready delta."""

    source_root = Path(source).resolve()
    target = Path(destination).resolve()
    model_config_path = Path(model_config).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    if target.exists():
        raise FileExistsError(target)
    files = _required_source_files(source_root)
    import yaml

    model = ModelSpec.model_validate(yaml.safe_load(model_config_path.read_text(encoding="utf-8")))
    if model.architecture != "inverse_llava":
        raise ValueError("champion conversion requires an inverse_llava model configuration")
    adapter_config = json.loads(files["adapter_config.json"].read_text(encoding="utf-8"))
    source_config = json.loads(files["config.json"].read_text(encoding="utf-8"))
    trainer_state = json.loads(files["trainer_state.json"].read_text(encoding="utf-8"))
    _validate_adapter_config(adapter_config, model)
    _validate_model_config(source_config, model)

    from safetensors.torch import load_file, save_file

    adapter = normalize_official_lora_state(load_file(files["adapter_model.safetensors"]))
    _validate_adapter_state(adapter, model)
    compact = _load_tensor_dict(files["non_lora_trainables.bin"])
    complete = _load_tensor_dict(files["non_lora_trainables_skip_layer.bin"])
    _validate_redundant_state(compact, complete)
    normalized_non_lora = normalize_champion_fusion_state(complete, model=model)
    fusion, copied_base = _partition_normalized_champion_state(normalized_non_lora)
    upstream_verification = _verify_upstream_projections(copied_base, model)
    overlap = set(adapter).intersection(fusion)
    if overlap:
        raise ValueError(f"adapter and fusion tensors collide: {sorted(overlap)[:8]}")
    combined = {**adapter, **fusion}

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        save_file(combined, temporary / MODEL_DELTA_FILENAME)
        source_files = {
            name: {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(files.items())
        }
        metadata = {
            "format": RELEASE_FORMAT,
            "source_format": CHAMPION_SOURCE_FORMAT,
            "model_delta_sha256": sha256_file(temporary / MODEL_DELTA_FILENAME),
            "model_delta_size_bytes": (temporary / MODEL_DELTA_FILENAME).stat().st_size,
            "tensor_count": len(combined),
            "tensor_names": sorted(combined),
            "source_files": source_files,
            "source_model_config_sha256": sha256_file(model_config_path),
            "training_summary": _training_summary(trainer_state),
            "base_model": model.language.checkpoint,
            "base_model_revision": model.language.revision,
            "vision_model": model.vision.checkpoint,
            "vision_model_revision": model.vision.revision,
            "upstream_base_projection_verification": upstream_verification,
        }
        atomic_write_json(temporary / "metadata.json", metadata)
        atomic_write_json(
            temporary / RELEASE_CONFIG_FILENAME,
            {
                "format": RELEASE_FORMAT,
                "model": model.model_dump(mode="json"),
                "weights": MODEL_DELTA_FILENAME,
                "metadata": "metadata.json",
                "conversation_template": "vicuna_v1",
            },
        )
        if model_card is not None:
            card = Path(model_card).resolve()
            if not card.is_file():
                raise FileNotFoundError(card)
            shutil.copyfile(card, temporary / "README.md")
        atomic_write_text(temporary / "COMPLETE", "complete\n")
        checksum_lines = [
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(temporary.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.name != "checksums.sha256"
        ]
        atomic_write_text(temporary / "checksums.sha256", "\n".join(checksum_lines) + "\n")
        os.chmod(temporary, 0o755)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target
