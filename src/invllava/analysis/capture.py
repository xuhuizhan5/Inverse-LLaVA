from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import Any

import torch
from torch import Tensor, nn


class ActivationCapture(AbstractContextManager["ActivationCapture"]):
    def __init__(
        self,
        model: nn.Module,
        module_names: Iterable[str],
        *,
        capture_inputs: Iterable[str] = (),
        to_cpu: bool = True,
    ) -> None:
        modules = dict(model.named_modules())
        missing = set(module_names) - modules.keys()
        if missing:
            raise ValueError(f"capture modules not found: {sorted(missing)}")
        self.modules = {name: modules[name] for name in module_names}
        self.to_cpu = to_cpu
        self.capture_inputs = set(capture_inputs)
        unknown_inputs = self.capture_inputs - self.modules.keys()
        if unknown_inputs:
            raise ValueError(f"input-capture modules not found: {sorted(unknown_inputs)}")
        self.values: dict[str, list[Tensor]] = {name: [] for name in self.modules}
        self.handles: list[Any] = []

    def __enter__(self) -> ActivationCapture:
        for name, module in self.modules.items():
            self.handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str) -> Any:
        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            value = _inputs[0] if name in self.capture_inputs else output
            value = value[0] if isinstance(value, tuple) else value
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"captured output at {name} is not a tensor")
            value = value.detach()
            self.values[name].append(value.cpu() if self.to_cpu else value)

        return capture

    def __exit__(self, *exc: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
