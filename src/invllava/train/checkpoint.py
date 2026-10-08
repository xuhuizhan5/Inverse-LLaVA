from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn

from invllava.artifacts.atomic import atomic_write_json, atomic_write_text
from invllava.artifacts.hashing import sha256_file
from invllava.train.state import TrainState, capture_rng_state, restore_rng_state

MODEL_DELTA_FILENAME = "model_delta.safetensors"
PORTABLE_CHECKPOINT_FILES = (
    "COMPLETE",
    "metadata.json",
    MODEL_DELTA_FILENAME,
    "state.json",
    "training_state.pt",
)
DEEPSPEED_CHECKPOINT_FILES = (
    "COMPLETE",
    "metadata.json",
    MODEL_DELTA_FILENAME,
    "state.json",
    "runtime",
)


def checkpoint_inventory(root: str | Path) -> dict[str, Any]:
    """Hash every complete checkpoint while rejecting incomplete file sets."""

    root = Path(root)
    entries: list[dict[str, Any]] = []
    for checkpoint in sorted(root.glob("step-*")):
        if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
            continue
        required = (
            DEEPSPEED_CHECKPOINT_FILES
            if (checkpoint / "runtime").is_dir()
            else PORTABLE_CHECKPOINT_FILES
        )
        missing = [name for name in required if not (checkpoint / name).exists()]
        if missing:
            raise ValueError(f"complete checkpoint {checkpoint} lacks files: {missing}")
        paths = [path for path in checkpoint.rglob("*") if path.is_file()]
        files = {}
        for path in sorted(paths):
            relative = path.relative_to(checkpoint).as_posix()
            files[relative] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        entries.append({"name": checkpoint.name, "files": files})
    if not entries:
        raise ValueError(f"no complete checkpoints found under {root}")
    return {"schema_version": 1, "checkpoints": entries}


class CheckpointManager:
    """Complete-marker checkpoints prevent resuming a partially written state."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        state: TrainState,
        metadata: dict[str, Any],
        rng_states: list[dict[str, Any]] | None = None,
    ) -> Path:
        destination = self.root / name
        if destination.exists():
            raise FileExistsError(destination)
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self.root))
        try:
            from safetensors.torch import save_file

            names = checkpoint_parameter_names(model)
            trainable = {
                key: value.detach().cpu().contiguous()
                for key, value in model.state_dict().items()
                if key in names
            }
            save_file(trainable, temporary / MODEL_DELTA_FILENAME)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_by_rank": rng_states or [capture_rng_state()],
                },
                temporary / "training_state.pt",
            )
            atomic_write_json(temporary / "state.json", state.to_dict())
            atomic_write_json(temporary / "metadata.json", metadata)
            atomic_write_text(temporary / "COMPLETE", "complete\n")
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def begin_distributed(self, *, name: str) -> Path:
        """Create an incomplete shared checkpoint directory on the main rank."""

        destination = self.root / name
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir()
        return destination

    def finalize_distributed(
        self,
        *,
        path: str | Path,
        model: nn.Module,
        model_state: dict[str, torch.Tensor],
        state: TrainState,
        metadata: dict[str, Any],
    ) -> None:
        """Publish compact scientific files after every rank saved runtime state."""

        destination = Path(path)
        if (destination / "COMPLETE").exists():
            raise FileExistsError(f"checkpoint is already complete: {destination}")
        if not (destination / "runtime").is_dir():
            raise FileNotFoundError(destination / "runtime")
        from safetensors.torch import save_file

        names = checkpoint_parameter_names(model)
        trainable = {
            key: value.detach().cpu().contiguous()
            for key, value in model_state.items()
            if key in names
        }
        if not trainable:
            raise ValueError("distributed checkpoint contains no learned tensors")
        save_file(trainable, destination / MODEL_DELTA_FILENAME)
        atomic_write_json(destination / "state.json", state.to_dict())
        atomic_write_json(destination / "metadata.json", metadata)
        atomic_write_text(destination / "COMPLETE", "complete\n")

    @staticmethod
    def load_distributed_state(path: str | Path) -> TrainState:
        source = Path(path)
        if not (source / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {source}")
        if not (source / "runtime").is_dir():
            raise ValueError(f"distributed checkpoint has no runtime state: {source}")
        return TrainState(**json.loads((source / "state.json").read_text(encoding="utf-8")))

    def load(
        self,
        path: str | Path,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        rank: int = 0,
    ) -> TrainState:
        source = Path(path)
        if not (source / "COMPLETE").is_file():
            raise ValueError(f"incomplete checkpoint: {source}")
        from safetensors.torch import load_file

        incompatible = model.load_state_dict(load_file(source / MODEL_DELTA_FILENAME), strict=False)
        expected_names = checkpoint_parameter_names(model)
        missing_expected = expected_names.intersection(incompatible.missing_keys)
        if missing_expected or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch; missing expected={sorted(missing_expected)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        saved = torch.load(source / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng_states = saved.get("rng_by_rank")
        if rng_states is None:
            if rank != 0:
                raise RuntimeError("checkpoint has no per-rank RNG state for DDP resume")
            rng_states = [saved["rng"]]
        if rank >= len(rng_states):
            raise RuntimeError(
                f"checkpoint contains {len(rng_states)} RNG states but process rank is {rank}"
            )
        restore_rng_state(rng_states[rank])
        return TrainState(**json.loads((source / "state.json").read_text(encoding="utf-8")))

    def prune_unprotected(self, *, keep_last: int, protected: set[str]) -> tuple[str, ...]:
        """Delete only complete checkpoints directly under this manager's root."""

        if keep_last < 1:
            raise ValueError("keep_last must be positive")
        complete = sorted(
            (
                path
                for path in self.root.glob("step-*")
                if path.is_dir() and (path / "COMPLETE").is_file()
            ),
            key=lambda path: path.name,
        )
        ordinary = [path for path in complete if path.name not in protected]
        targets = ordinary[:-keep_last]
        removed: list[str] = []
        root = self.root.resolve()
        for target in targets:
            resolved = target.resolve()
            if resolved.parent != root or not resolved.name.startswith("step-"):
                raise ValueError(f"refusing checkpoint prune target: {resolved}")
            shutil.rmtree(resolved)
            removed.append(resolved.name)
        return tuple(removed)


def checkpoint_parameter_names(model: nn.Module) -> set[str]:
    """Persist learned adapters and every configuration-owned fusion tensor.

    Frozen base weights are reconstructed from their immutable Hub revision.
    Fusion and LoRA tensors have no such upstream source and therefore remain
    part of the scientific checkpoint when a mechanism ablation freezes them.
    """

    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        or any(marker in f".{name}." for marker in (".fusion.", ".lora_a.", ".lora_b."))
    }


def _is_checkpoint_state(name: str, model: nn.Module) -> bool:
    return name in checkpoint_parameter_names(model)


def load_trainable_weights(path: str | Path, model: nn.Module) -> None:
    """Initialize a new scientific run without restoring optimizer/RNG state."""

    source = Path(path)
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"incomplete initialization checkpoint: {source}")
    from safetensors.torch import load_file

    weights = load_file(source / MODEL_DELTA_FILENAME)
    incompatible = model.load_state_dict(weights, strict=False)
    loaded = set(weights)
    unexpected = set(incompatible.unexpected_keys)
    expected = checkpoint_parameter_names(model)
    missing = expected.intersection(incompatible.missing_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"initialization checkpoint mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if not loaded:
        raise ValueError("initialization checkpoint contains no trainable weights")


def _select_component_state(
    weights: dict[str, torch.Tensor], *, prefix: str
) -> dict[str, torch.Tensor]:
    normalized = prefix if prefix.endswith(".") else prefix + "."
    selected = {
        name.removeprefix(normalized): tensor
        for name, tensor in weights.items()
        if name.startswith(normalized)
    }
    if not selected:
        raise ValueError(f"checkpoint contains no tensors under component prefix {prefix}")
    return selected


def load_component_weights(path: str | Path, module: nn.Module, *, prefix: str) -> None:
    """Strictly load one component from a complete multimodal delta."""

    source = Path(path)
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"incomplete component checkpoint: {source}")
    from safetensors.torch import load_file

    weights = _select_component_state(load_file(source / MODEL_DELTA_FILENAME), prefix=prefix)
    incompatible = module.load_state_dict(weights, strict=False)
    expected = checkpoint_parameter_names(module)
    missing = expected.intersection(incompatible.missing_keys)
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"component checkpoint mismatch; missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
