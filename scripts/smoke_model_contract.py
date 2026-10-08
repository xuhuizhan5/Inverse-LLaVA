#!/usr/bin/env python3
"""No-download end-to-end model/checkpoint smoke test for a staged GPU host."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from invllava.config.schema import OptimizerSpec
from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.generation import generate
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.modeling import InverseLLaVAForConditionalGeneration
from invllava.model.vision import VisionEncoder
from invllava.train.checkpoint import CheckpointManager
from invllava.train.optimizer import build_scheduler
from invllava.train.state import TrainState


class TinyVisionTower(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, feature_dim, bias=False)

    def forward(self, *, pixel_values: torch.Tensor, output_hidden_states: bool) -> object:
        if not output_hidden_states:
            raise ValueError("vision contract requires hidden states")
        patches = pixel_values.flatten(2).transpose(1, 2)
        patches = self.projection(patches)
        cls = patches.new_zeros((patches.shape[0], 1, patches.shape[-1]))
        return SimpleNamespace(hidden_states=(torch.cat((cls, patches), dim=1),))


def build_tiny_model(device: torch.device) -> InverseLLaVAForConditionalGeneration:
    feature_dim = 8
    architecture = LlamaArchitecture(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    fusion = FusionBlock(
        hidden_size=architecture.hidden_size,
        visual_size=feature_dim,
        target_sizes={"q": 16, "k": 16, "v": 16},
    )
    language = InverseLlamaForCausalLM(architecture, fusion_layers={0: fusion})
    language.requires_grad_(False)
    fusion.requires_grad_(True)
    vision = VisionEncoder(TinyVisionTower(feature_dim), feature_layers=(-1,), freeze=True)
    model = InverseLLaVAForConditionalGeneration(
        language,
        vision,
        image_token_id=-200,
        visual_feature_dim=feature_dim,
        max_length=32,
    )
    return model.to(device)


def _train_step(
    model: InverseLLaVAForConditionalGeneration,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    device: torch.device,
) -> tuple[float, int]:
    """Run one stochastic step so RNG restoration affects continuation parity."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pixels = torch.randn(1, 3, 2, 2, device=device)
    features = model.encode_images(pixels)
    input_ids = torch.tensor([[1, -200, 5, 6]], device=device)
    output = model(input_ids=input_ids, image_features=[[features[0]]], labels=input_ids.clone())
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("tiny multimodal forward produced an invalid loss")
    output.loss.backward()
    fusion_with_grad = sum(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if ".fusion." in f".{name}." and parameter.requires_grad
    )
    if fusion_with_grad == 0:
        raise RuntimeError("fusion pathway did not receive gradients")
    optimizer.step()
    scheduler.step()
    return float(output.loss.detach()), fusion_with_grad


def _generate_fixture(
    model: InverseLLaVAForConditionalGeneration, *, device: torch.device
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pixels = torch.full((1, 3, 2, 2), 0.25, device=device)
        features = model.encode_images(pixels)
        input_ids = torch.tensor([[1, -200, 5, 6]], device=device)
        prepared = model.prepare(input_ids=input_ids, image_features=[[features[0]]])
        generated = generate(
            model.language_model,
            prepared,
            max_new_tokens=2,
            eos_token_id=2,
            temperature=0.0,
        )
    model.train(was_training)
    return generated.detach().cpu()


def _trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _assert_tree_close(actual: Any, expected: Any, *, path: str = "root") -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise AssertionError(f"{path}: expected tensor, got {type(actual).__name__}")
        torch.testing.assert_close(actual.detach().cpu(), expected, rtol=1e-6, atol=1e-7)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: mapping keys differ")
        for key in expected:
            _assert_tree_close(actual[key], expected[key], path=f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError(f"{path}: sequence structure differs")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected, strict=True)):
            _assert_tree_close(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested but CUDA is unavailable")
    torch.manual_seed(42)
    model = build_tiny_model(device)
    fusion_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if ".fusion." in f".{name}." and parameter.requires_grad
    ]
    if not fusion_parameters:
        raise RuntimeError("tiny model contains no trainable fusion parameters")
    optimizer = torch.optim.AdamW(fusion_parameters, lr=1e-3)
    scheduler = build_scheduler(
        optimizer,
        OptimizerSpec(learning_rate=1e-3, warmup_ratio=0.0),
        total_steps=2,
    )
    first_loss, fusion_parameters_with_grad = _train_step(
        model, optimizer, scheduler, device=device
    )
    manager = CheckpointManager(output / "checkpoints")
    checkpoint = manager.save(
        name="step-0000001",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainState(global_step=1, samples_seen=1),
        metadata={"contract": "tiny-multimodal-no-download"},
    )
    checkpoint_generation = _generate_fixture(model, device=device)

    second_loss, _ = _train_step(model, optimizer, scheduler, device=device)
    uninterrupted_model = _trainable_state(model)
    uninterrupted_optimizer = _clone_tree(optimizer.state_dict())
    uninterrupted_scheduler = _clone_tree(scheduler.state_dict())

    restored = manager.load(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    reloaded_generation = _generate_fixture(model, device=device)
    torch.testing.assert_close(reloaded_generation, checkpoint_generation)
    resumed_second_loss, _ = _train_step(model, optimizer, scheduler, device=device)
    resumed_model = _trainable_state(model)
    _assert_tree_close(resumed_model, uninterrupted_model, path="model")
    _assert_tree_close(optimizer.state_dict(), uninterrupted_optimizer, path="optimizer")
    _assert_tree_close(scheduler.state_dict(), uninterrupted_scheduler, path="scheduler")
    if abs(resumed_second_loss - second_loss) > 1e-7:
        raise AssertionError(
            f"resumed loss {resumed_second_loss} differs from uninterrupted loss {second_loss}"
        )
    result = {
        "device": str(device),
        "first_loss": first_loss,
        "second_loss": second_loss,
        "resumed_second_loss": resumed_second_loss,
        "generated_shape": list(checkpoint_generation.shape),
        "fusion_parameters_with_grad": fusion_parameters_with_grad,
        "optimizer_state_entries": len(optimizer.state),
        "restored_global_step": restored.global_step,
        "checkpoint_generation_parity": True,
        "exact_resume_model_optimizer_scheduler_rng_parity": True,
        "checkpoint": str(checkpoint),
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
