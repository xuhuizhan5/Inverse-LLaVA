from pathlib import Path
from unittest.mock import patch

import torch

from invllava.config.schema import RuntimeSpec
from invllava.runtime.numerics import configure_torch_numerics


def _runtime(*, allow_tf32: bool, deterministic: bool) -> RuntimeSpec:
    return RuntimeSpec(
        id="numerics-test",
        accelerator="cpu",
        mixed_precision="no",
        num_processes=1,
        dataloader_workers=0,
        distributed_strategy="none",
        attention_backend="eager",
        allow_tf32=allow_tf32,
        deterministic_algorithms=deterministic,
        cudnn_benchmark=False,
        run_root=Path("/tmp/invllava-test-runs"),
        cache_root=Path("/tmp/invllava-test-cache"),
        tracker="local",
        keep_last_checkpoints=1,
    )


def test_numerical_policy_uses_public_matmul_precision_api() -> None:
    original_precision = torch.get_float32_matmul_precision()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    convolution = getattr(torch.backends.cudnn, "conv", None)
    uses_new_cudnn_api = convolution is not None and hasattr(convolution, "fp32_precision")
    original_cudnn_precision = (
        convolution.fp32_precision if uses_new_cudnn_api else torch.backends.cudnn.allow_tf32
    )
    try:
        with patch(
            "torch.set_float32_matmul_precision",
            wraps=torch.set_float32_matmul_precision,
        ) as set_precision:
            report = configure_torch_numerics(_runtime(allow_tf32=True, deterministic=False))
        set_precision.assert_called_once_with("high")
        assert report["allow_tf32"] is True
        assert report["float32_matmul_precision"] == "high"
        assert report["deterministic_algorithms"] is False
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.use_deterministic_algorithms(original_deterministic)
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        if uses_new_cudnn_api:
            convolution.fp32_precision = original_cudnn_precision
        else:
            torch.backends.cudnn.allow_tf32 = original_cudnn_precision


def test_disabling_tf32_selects_highest_matmul_precision() -> None:
    original_precision = torch.get_float32_matmul_precision()
    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_cudnn_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    convolution = getattr(torch.backends.cudnn, "conv", None)
    uses_new_cudnn_api = convolution is not None and hasattr(convolution, "fp32_precision")
    original_cudnn_precision = (
        convolution.fp32_precision if uses_new_cudnn_api else torch.backends.cudnn.allow_tf32
    )
    try:
        report = configure_torch_numerics(_runtime(allow_tf32=False, deterministic=True))
        assert report["float32_matmul_precision"] == "highest"
        assert report["deterministic_algorithms"] is True
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.use_deterministic_algorithms(original_deterministic)
        torch.backends.cudnn.benchmark = original_cudnn_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic
        if uses_new_cudnn_api:
            convolution.fp32_precision = original_cudnn_precision
        else:
            torch.backends.cudnn.allow_tf32 = original_cudnn_precision
