import torch

from invllava.config.schema import OptimizerSpec
from invllava.train.checkpoint import (
    CheckpointManager,
    _is_checkpoint_state,
    _select_component_state,
    checkpoint_inventory,
)
from invllava.train.engine import _restore_deepspeed_checkpoint
from invllava.train.optimizer import build_scheduler
from invllava.train.state import TrainState


def test_checkpoint_selection_includes_frozen_fusion() -> None:
    class ToyDelta(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.fusion = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Linear(2, 2, bias=False)
            self.lora_a = torch.nn.Linear(2, 1, bias=False)
            self.lora_b = torch.nn.Linear(1, 2, bias=False)
            self.base.requires_grad_(False)
            self.fusion.requires_grad_(False)
            self.lora_a.requires_grad_(False)
            self.lora_b.requires_grad_(False)

    model = ToyDelta()
    assert not _is_checkpoint_state("base.weight", model)
    assert _is_checkpoint_state("fusion.weight", model)
    assert _is_checkpoint_state("lora_a.weight", model)
    assert _is_checkpoint_state("lora_b.weight", model)
    assert _is_checkpoint_state("adapter.weight", model)


def test_component_state_strips_only_the_exact_prefix() -> None:
    state = {
        "language_model.model.adapter.weight": torch.ones(1),
        "multimodal_projector.weight": torch.zeros(1),
    }
    selected = _select_component_state(state, prefix="language_model")
    assert selected.keys() == {"model.adapter.weight"}


def test_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, OptimizerSpec(), total_steps=2)
    original = model.weight.detach().clone()
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save(
        name="step-1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainState(global_step=1, batch_in_epoch=1),
        metadata={"fixture": True},
    )
    with torch.no_grad():
        model.weight.zero_()
    restored = manager.load(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)
    torch.testing.assert_close(model.weight, original)
    assert restored.global_step == 1
    assert (checkpoint / "COMPLETE").is_file()


def test_checkpoint_preserves_frozen_fusion_but_not_upstream_base(tmp_path) -> None:
    class ToyDelta(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.fusion = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Linear(2, 2, bias=False)
            self.base.requires_grad_(False)
            self.fusion.requires_grad_(False)

    model = ToyDelta()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, OptimizerSpec(), total_steps=1)
    original_fusion = model.fusion.weight.detach().clone()
    original_adapter = model.adapter.weight.detach().clone()
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save(
        name="step-1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainState(global_step=1),
        metadata={"fixture": "frozen-fusion"},
    )
    with torch.no_grad():
        model.base.weight.fill_(7)
        model.fusion.weight.zero_()
        model.adapter.weight.zero_()
    manager.load(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)
    torch.testing.assert_close(model.fusion.weight, original_fusion)
    torch.testing.assert_close(model.adapter.weight, original_adapter)
    torch.testing.assert_close(model.base.weight, torch.full_like(model.base.weight, 7))


def test_distributed_checkpoint_requires_runtime_state_before_completion(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.begin_distributed(name="step-1")
    runtime = checkpoint / "runtime" / "pytorch_model"
    runtime.mkdir(parents=True)
    (runtime / "rank0-state.pt").write_bytes(b"runtime-state")

    manager.finalize_distributed(
        path=checkpoint,
        model=model,
        model_state=model.state_dict(),
        state=TrainState(global_step=1),
        metadata={"runtime_state_format": "accelerate-deepspeed-zero2"},
    )

    restored = manager.load_distributed_state(checkpoint)
    inventory = checkpoint_inventory(tmp_path)
    assert restored.global_step == 1
    assert inventory["checkpoints"][0]["name"] == "step-1"
    assert "runtime/pytorch_model/rank0-state.pt" in inventory["checkpoints"][0]["files"]


def test_distributed_resume_loads_delta_then_runtime_nonstrict(tmp_path) -> None:
    class FakeAccelerator:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        @staticmethod
        def unwrap_model(model):
            return model

        def load_state(self, path: str, *, load_module_strict: bool) -> None:
            self.calls.append((path, load_module_strict))

    model = torch.nn.Linear(3, 2)
    expected = model.weight.detach().clone()
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.begin_distributed(name="step-1")
    runtime = checkpoint / "runtime" / "pytorch_model"
    runtime.mkdir(parents=True)
    (runtime / "rank0-state.pt").write_bytes(b"runtime-state")
    manager.finalize_distributed(
        path=checkpoint,
        model=model,
        model_state=model.state_dict(),
        state=TrainState(global_step=1),
        metadata={"runtime_state_format": "accelerate-deepspeed-zero2"},
    )
    with torch.no_grad():
        model.weight.zero_()
    accelerator = FakeAccelerator()

    restored = _restore_deepspeed_checkpoint(
        accelerator=accelerator,
        model=model,
        source=checkpoint,
    )

    torch.testing.assert_close(model.weight, expected)
    assert restored.global_step == 1
    assert accelerator.calls == [(str(checkpoint / "runtime"), False)]
