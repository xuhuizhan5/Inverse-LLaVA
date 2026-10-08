"""Optional TorchInductor/Triton compilation with explicit provenance.

The eager module and its state dictionary remain authoritative. Compilation is
applied in place through ``nn.Module.compile`` so checkpoint parameter names do
not acquire wrapper prefixes. The first real call performs lazy compilation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from invllava.config.schema import KernelOptimizationSpec
from invllava.model.fusion import FusionBlock


@dataclass(frozen=True)
class KernelOptimizationReport:
    enabled: bool
    compile_scope: str
    compile_backend: str | None
    compile_mode: str | None
    dynamic_shapes: bool
    fullgraph: bool
    triton_version: str | None
    compiled_modules: tuple[str, ...]
    state_keys_preserved: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _triton_version() -> str | None:
    try:
        import triton
    except ImportError:
        return None
    value = getattr(triton, "__version__", None)
    return str(value) if value is not None else "unknown"


def _compilation_targets(model: nn.Module, scope: str) -> list[tuple[str, nn.Module]]:
    if scope == "fusion":
        return [
            (name or "<root-fusion>", module)
            for name, module in model.named_modules()
            if isinstance(module, FusionBlock)
        ]
    if scope == "language_model":
        language_model = getattr(model, "language_model", model)
        if not isinstance(language_model, nn.Module):
            raise TypeError("model.language_model is not a torch module")
        return [("language_model" if language_model is not model else "<root>", language_model)]
    raise ValueError(f"unsupported compile scope: {scope}")


def configure_model_kernels(
    model: nn.Module,
    spec: KernelOptimizationSpec,
) -> KernelOptimizationReport:
    """Apply an explicitly requested compile mode or return a no-op report.

    There is no silent fallback. An unavailable CUDA/Triton/compiler path raises
    and the caller can rerun with the portable runtime configuration.
    """

    existing = getattr(model, "_invllava_kernel_optimization_report", None)
    if existing is not None:
        if not isinstance(existing, KernelOptimizationReport):
            raise RuntimeError("model contains an invalid kernel-optimization marker")
        return existing
    if spec.compile_scope == "none":
        report = KernelOptimizationReport(
            enabled=False,
            compile_scope="none",
            compile_backend=None,
            compile_mode=None,
            dynamic_shapes=False,
            fullgraph=False,
            triton_version=None,
            compiled_modules=(),
            state_keys_preserved=True,
        )
        model._invllava_kernel_optimization_report = report  # type: ignore[attr-defined]
        return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "kernel-optimized runtime requires a visible CUDA device; use the portable runtime"
        )
    triton_version = _triton_version()
    if spec.require_triton and triton_version is None:
        raise RuntimeError(
            "kernel-optimized runtime requires the CUDA build's compatible Triton package; "
            "use the portable runtime instead of installing an arbitrary Triton version"
        )
    targets = _compilation_targets(model, spec.compile_scope)
    if not targets:
        raise RuntimeError(f"compile scope {spec.compile_scope!r} selected no modules")
    state_keys_before = tuple(model.state_dict())
    for name, module in targets:
        compile_method = getattr(module, "compile", None)
        if not callable(compile_method):
            raise RuntimeError(
                f"PyTorch {torch.__version__} does not expose nn.Module.compile for {name}"
            )
        compile_method(
            backend=spec.compile_backend,
            mode=spec.compile_mode,
            dynamic=spec.dynamic_shapes,
            fullgraph=spec.fullgraph,
        )
    state_keys_preserved = tuple(model.state_dict()) == state_keys_before
    if not state_keys_preserved:
        raise RuntimeError("kernel compilation changed checkpoint state-dictionary keys")
    report = KernelOptimizationReport(
        enabled=True,
        compile_scope=spec.compile_scope,
        compile_backend=spec.compile_backend,
        compile_mode=spec.compile_mode,
        dynamic_shapes=spec.dynamic_shapes,
        fullgraph=spec.fullgraph,
        triton_version=triton_version,
        compiled_modules=tuple(name for name, _ in targets),
        state_keys_preserved=state_keys_preserved,
    )
    model._invllava_kernel_optimization_report = report  # type: ignore[attr-defined]
    return report
