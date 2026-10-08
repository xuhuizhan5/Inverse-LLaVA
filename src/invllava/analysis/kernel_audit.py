"""Numerical-parity and warm-runtime audit for optional compiled execution."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from invllava.analysis.profiling import profile_callable
from invllava.artifacts.hashing import sha256_file
from invllava.config.schema import RuntimeSpec
from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.generation import generate
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.types import ExpandedSequence, FusionState
from invllava.runtime.optimization import configure_model_kernels


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_once(operation: Any, device: torch.device) -> tuple[Any, float]:
    _synchronize(device)
    start = time.perf_counter()
    value = operation()
    _synchronize(device)
    return value, time.perf_counter() - start


def _error(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    left = reference.detach().float()
    right = candidate.detach().float()
    absolute = (left - right).abs()
    denominator = left.abs().clamp_min(1e-7)
    return {
        "max_absolute": float(absolute.max().item()),
        "mean_absolute": float(absolute.mean().item()),
        "max_relative": float((absolute / denominator).max().item()),
    }


def _tiny_models(
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    visual_size: int,
    layers: int,
) -> tuple[
    InverseLlamaForCausalLM,
    InverseLlamaForCausalLM,
    dict[str, Tensor | FusionState],
]:
    if hidden_size % 4:
        raise ValueError("hidden size must be divisible by four")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    architecture = LlamaArchitecture(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max(128, 2 * sequence_length),
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=127,
    )
    target_sizes = {
        "q": hidden_size,
        "k": hidden_size // 2,
        "v": hidden_size // 2,
    }
    fusion = FusionBlock(
        hidden_size=hidden_size,
        visual_size=visual_size,
        target_sizes=target_sizes,
        targets=("q", "k", "v"),
        operator="concat",
        output_init_std=1e-3,
    )
    reference = InverseLlamaForCausalLM(
        architecture,
        fusion_layers={0: fusion},
        attention_backend="sdpa",
    ).to(device=device, dtype=dtype)
    candidate = copy.deepcopy(reference)
    input_ids = torch.randint(
        3,
        architecture.vocab_size - 1,
        (batch_size, sequence_length),
        device=device,
    )
    labels = input_ids.clone()
    visual = torch.randn(
        batch_size,
        sequence_length,
        visual_size,
        device=device,
        dtype=dtype,
    )
    text_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)
    text_mask[:, ::2] = True
    vision_mask = ~text_mask
    state = FusionState(visual, text_mask, vision_mask)
    return reference, candidate, {"input_ids": input_ids, "labels": labels, "state": state}


def _forward(model: InverseLlamaForCausalLM, inputs: dict[str, Any]) -> Any:
    return model(
        input_ids=inputs["input_ids"],
        labels=inputs["labels"],
        fusion_state=inputs["state"],
    )


def _greedy_tokens(
    model: InverseLlamaForCausalLM,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
) -> Tensor:
    input_ids = inputs["input_ids"]
    embeddings = model.model.embed_tokens(input_ids)
    initial = ExpandedSequence(
        inputs_embeds=embeddings,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        position_ids=torch.arange(input_ids.shape[1], device=input_ids.device)
        .unsqueeze(0)
        .expand_as(input_ids),
        labels=None,
        fusion_state=inputs["state"],
    )
    return generate(
        model,
        initial,
        max_new_tokens=max_new_tokens,
        eos_token_id=model.config.eos_token_id,
        temperature=0.0,
    )


def _fusion_gradients(model: InverseLlamaForCausalLM) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if ".fusion." in name and parameter.requires_grad and parameter.grad is not None
    }


def _export_trace(model: InverseLlamaForCausalLM, inputs: dict[str, Any], path: Path) -> str:
    if path.exists():
        raise FileExistsError(path)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        acc_events=True,
    ) as profiler:
        with torch.inference_mode(), torch.profiler.record_function("invllava::kernel_audit"):
            _forward(model, inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(path))
    return sha256_file(path)


def run_kernel_audit(
    runtime: RuntimeSpec,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    seed: int = 2026,
    batch_size: int = 2,
    sequence_length: int = 32,
    hidden_size: int = 64,
    visual_size: int = 32,
    layers: int = 2,
    warmups: int = 5,
    repetitions: int = 15,
    atol: float | None = None,
    rtol: float | None = None,
    trace: str | Path | None = None,
) -> dict[str, Any]:
    """Compare portable and candidate runtimes on one identical tiny model.

    This is a no-download qualification test, not an estimate of 7B-model
    throughput. A candidate passes only when forward values, trainable fusion
    gradients, and deterministic greedy tokens meet the declared contract.
    """

    if warmups < 0 or repetitions < 3:
        raise ValueError("kernel audit requires non-negative warmups and >=3 repetitions")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA kernel audit but no CUDA device is visible")
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    if target.type == "cpu" and torch_dtype != torch.float32:
        raise ValueError("CPU kernel audits use float32")
    tolerance = 1e-5 if torch_dtype == torch.float32 else 3e-2
    absolute_tolerance = tolerance if atol is None else atol
    relative_tolerance = tolerance if rtol is None else rtol
    reference, candidate, inputs = _tiny_models(
        device=target,
        dtype=torch_dtype,
        seed=seed,
        batch_size=batch_size,
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        visual_size=visual_size,
        layers=layers,
    )
    state_keys_before = tuple(candidate.state_dict())
    report = configure_model_kernels(candidate, runtime.kernel_optimization)
    state_keys_after = tuple(candidate.state_dict())

    reference.eval()
    candidate.eval()
    with torch.inference_mode():
        reference_output, reference_cold = _timed_once(lambda: _forward(reference, inputs), target)
        candidate_output, candidate_cold = _timed_once(lambda: _forward(candidate, inputs), target)
        reference_tokens = _greedy_tokens(reference, inputs, max_new_tokens=4)
        candidate_tokens = _greedy_tokens(candidate, inputs, max_new_tokens=4)
    logits_error = _error(reference_output.logits, candidate_output.logits)
    loss_error = _error(reference_output.loss, candidate_output.loss)
    inference_close = torch.allclose(
        reference_output.logits,
        candidate_output.logits,
        atol=absolute_tolerance,
        rtol=relative_tolerance,
    )

    reference.train()
    candidate.train()
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    reference_train = _forward(reference, inputs)
    candidate_train = _forward(candidate, inputs)
    if reference_train.loss is None or candidate_train.loss is None:
        raise RuntimeError("kernel audit training path returned no loss")
    reference_train.loss.backward()
    candidate_train.loss.backward()
    reference_gradients = _fusion_gradients(reference)
    candidate_gradients = _fusion_gradients(candidate)
    gradient_keys_match = reference_gradients.keys() == candidate_gradients.keys()
    gradient_errors = {
        name: _error(reference_gradients[name], candidate_gradients[name])
        for name in reference_gradients.keys() & candidate_gradients.keys()
    }
    gradients_close = gradient_keys_match and all(
        torch.allclose(
            reference_gradients[name],
            candidate_gradients[name],
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )
        for name in reference_gradients.keys() & candidate_gradients.keys()
    )

    reference.eval()
    candidate.eval()
    portable_timing = profile_callable(
        lambda: _forward(reference, inputs), warmups=warmups, repetitions=repetitions
    )
    candidate_timing = profile_callable(
        lambda: _forward(candidate, inputs), warmups=warmups, repetitions=repetitions
    )
    trace_payload = None
    if trace is not None:
        trace_path = Path(trace)
        trace_payload = {
            "path": str(trace_path),
            "sha256": _export_trace(candidate, inputs, trace_path),
        }
    state_keys_preserved = state_keys_before == state_keys_after
    greedy_exact = torch.equal(reference_tokens, candidate_tokens)
    passed = bool(state_keys_preserved and inference_close and gradients_close and greedy_exact)
    hardware: dict[str, Any] = {
        "device": str(target),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if target.type == "cuda":
        index = target.index if target.index is not None else torch.cuda.current_device()
        hardware.update(
            {
                "name": torch.cuda.get_device_name(index),
                "compute_capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "schema_version": 1,
        "measurement": "measured_tiny_model_qualification_not_7b_throughput",
        "passed": passed,
        "runtime_id": runtime.id,
        "kernel_optimization": report.to_dict(),
        "hardware": hardware,
        "fixture": {
            "seed": seed,
            "dtype": dtype,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "visual_size": visual_size,
            "layers": layers,
        },
        "tolerances": {"atol": absolute_tolerance, "rtol": relative_tolerance},
        "parity": {
            "state_keys_preserved": state_keys_preserved,
            "inference_allclose": bool(inference_close),
            "greedy_tokens_exact": greedy_exact,
            "gradient_keys_match": gradient_keys_match,
            "fusion_gradients_allclose": gradients_close,
            "logits_error": logits_error,
            "loss_error": loss_error,
            "fusion_gradient_errors": gradient_errors,
        },
        "cold_start_seconds": {
            "portable": reference_cold,
            "candidate_including_lazy_compile": candidate_cold,
        },
        "warm_timing": {
            "portable": portable_timing.to_dict(),
            "candidate": candidate_timing.to_dict(),
            "speedup": portable_timing.median_seconds / candidate_timing.median_seconds,
        },
        "trace": trace_payload,
    }
