"""One explicit numerical policy for training and native inference."""

from __future__ import annotations

import os
from typing import Any

from invllava.config.schema import RuntimeSpec


def configure_torch_numerics_policy(
    *,
    mixed_precision: str,
    allow_tf32: bool,
    deterministic_algorithms: bool,
    cudnn_benchmark: bool,
) -> dict[str, Any]:
    """Apply and report an explicit PyTorch numerical policy.

    ``set_float32_matmul_precision`` is the current public PyTorch interface
    for permitting TF32-like internal computation of FP32 matrix products.  A
    compatibility assignment is retained only for cuDNN convolution because
    the CLIP patch embedding is a convolution and PyTorch versions expose its
    policy through different attributes.
    """

    import torch

    matmul_precision = "high" if allow_tf32 else "highest"
    torch.set_float32_matmul_precision(matmul_precision)
    torch.use_deterministic_algorithms(deterministic_algorithms)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = deterministic_algorithms

    convolution = getattr(torch.backends.cudnn, "conv", None)
    if convolution is not None and hasattr(convolution, "fp32_precision"):
        convolution.fp32_precision = "tf32" if allow_tf32 else "ieee"
        cudnn_interface = "cudnn.conv.fp32_precision"
    else:
        torch.backends.cudnn.allow_tf32 = allow_tf32
        cudnn_interface = "cudnn.allow_tf32"

    return {
        "mixed_precision": mixed_precision,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_policy_interface": cudnn_interface,
        "cuda_allocator_conf": os.environ.get("PYTORCH_ALLOC_CONF"),
    }


def configure_torch_numerics(runtime: RuntimeSpec) -> dict[str, Any]:
    """Apply the numerical fields from a resolved runtime configuration."""

    return configure_torch_numerics_policy(
        mixed_precision=runtime.mixed_precision,
        allow_tf32=runtime.allow_tf32,
        deterministic_algorithms=runtime.deterministic_algorithms,
        cudnn_benchmark=runtime.cudnn_benchmark,
    )
