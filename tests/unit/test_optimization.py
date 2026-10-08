from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

from invllava.analysis.kernel_audit import run_kernel_audit
from invllava.config.schema import KernelOptimizationSpec, RuntimeSpec
from invllava.model.fusion import FusionBlock
from invllava.runtime.optimization import configure_model_kernels


class RecordingFusion(FusionBlock):
    def __init__(self) -> None:
        super().__init__(
            hidden_size=8,
            visual_size=4,
            target_sizes={"q": 8, "k": 8, "v": 8},
        )
        self.compile_arguments: dict[str, object] | None = None

    def compile(self, **kwargs: object) -> None:
        self.compile_arguments = kwargs


def _runtime(optimization: KernelOptimizationSpec) -> RuntimeSpec:
    return RuntimeSpec(
        id="test-runtime",
        accelerator="cpu" if optimization.compile_scope == "none" else "cuda",
        mixed_precision="no",
        num_processes=1,
        dataloader_workers=0,
        distributed_strategy="none",
        attention_backend="sdpa",
        run_root=Path("/tmp/invllava-test-runs"),
        cache_root=Path("/tmp/invllava-test-cache"),
        tracker="local",
        keep_last_checkpoints=1,
        kernel_optimization=optimization,
    )


def test_disabled_optimization_is_a_true_no_op() -> None:
    model = nn.Linear(3, 2)
    keys = tuple(model.state_dict())
    with patch(
        "invllava.runtime.optimization._triton_version",
        side_effect=AssertionError("disabled runtime imported Triton"),
    ):
        report = configure_model_kernels(model, KernelOptimizationSpec())
    assert not report.enabled
    assert tuple(model.state_dict()) == keys


def test_fusion_scope_compiles_only_fusion_and_preserves_state_keys() -> None:
    model = nn.Sequential(nn.Linear(8, 8), RecordingFusion())
    keys = tuple(model.state_dict())
    spec = KernelOptimizationSpec(
        compile_scope="fusion",
        compile_mode="reduce-overhead",
        dynamic_shapes=True,
    )
    with (
        patch("invllava.runtime.optimization.torch.cuda.is_available", return_value=True),
        patch("invllava.runtime.optimization._triton_version", return_value="fixture"),
    ):
        report = configure_model_kernels(model, spec)
    fusion = model[1]
    assert isinstance(fusion, RecordingFusion)
    assert fusion.compile_arguments == {
        "backend": "inductor",
        "mode": "reduce-overhead",
        "dynamic": True,
        "fullgraph": False,
    }
    assert report.compiled_modules == ("1",)
    assert tuple(model.state_dict()) == keys


def test_disabled_scope_rejects_non_neutral_compile_options() -> None:
    with pytest.raises(ValueError, match="neutral"):
        KernelOptimizationSpec(compile_scope="none", compile_mode="max-autotune")


def test_portable_kernel_audit_exercises_parity_contract_on_cpu() -> None:
    result = run_kernel_audit(
        _runtime(KernelOptimizationSpec()),
        device="cpu",
        dtype="float32",
        batch_size=1,
        sequence_length=4,
        hidden_size=16,
        visual_size=8,
        layers=1,
        warmups=0,
        repetitions=3,
    )
    assert result["passed"]
    assert result["parity"]["greedy_tokens_exact"]
    assert not result["kernel_optimization"]["enabled"]
