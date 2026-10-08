from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from invllava.analysis.capture import ActivationCapture
from invllava.artifacts.atomic import atomic_write_json
from invllava.artifacts.hashing import sha256_file
from invllava.eval.runtime import InverseGenerator
from invllava.eval.types import EvaluationExample, generation_request


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if value.ndim != 3 or mask.shape != value.shape[:2]:
        raise ValueError("pooled representation and mask do not align")
    weights = mask.unsqueeze(-1).to(value.dtype)
    return (value * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def _last_valid(value: Tensor, mask: Tensor) -> Tensor:
    """Select the final valid prompt position from a padded hidden-state batch."""

    if value.ndim != 3 or mask.shape != value.shape[:2]:
        raise ValueError("hidden states and attention mask do not align")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).amax(dim=1)
    if bool((indices < 0).any()):
        raise ValueError("each representation row must contain a valid prompt token")
    return value[torch.arange(value.shape[0], device=value.device), indices]


def _append(storage: dict[str, list[np.ndarray]], key: str, value: Tensor) -> None:
    storage.setdefault(key, []).append(value.detach().float().cpu().numpy())


@torch.inference_mode()
def capture_representations(
    generator: InverseGenerator,
    examples: Sequence[EvaluationExample],
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Capture sample-level arrays under one explicit mean-pooling policy.

    The study intentionally requires one image per example. This keeps native
    image rows aligned with sample IDs and avoids silently averaging unrelated
    multi-image panels.
    """

    if batch_size <= 0 or not examples:
        raise ValueError("representation capture requires examples and a positive batch size")
    if any(len(example.images) != 1 for example in examples):
        raise ValueError("representation capture currently requires exactly one image per sample")
    model = generator.model
    modules = dict(model.named_modules())
    module_names = ["vision_encoder"]
    capture_inputs: list[str] = []
    mapper_modules: dict[tuple[int, str], str] = {}
    joint_modules: dict[tuple[int, str], str] = {}
    if hasattr(model, "multimodal_projector"):
        module_names.append("multimodal_projector")
    fusion_layers = [
        index
        for index, layer in enumerate(model.language_model.model.layers)
        if layer.self_attn.fusion is not None
    ]
    for layer in fusion_layers:
        fusion = model.language_model.model.layers[layer].self_attn.fusion
        assert fusion is not None
        for target in fusion.targets:
            mapper = f"language_model.model.layers.{layer}.self_attn.fusion.text_to_vision.{target}"
            joint = f"language_model.model.layers.{layer}.self_attn.fusion.output.{target}"
            if mapper not in modules or joint not in modules:
                raise ValueError(
                    f"fusion capture modules are missing at layer {layer}, target {target}"
                )
            module_names.extend((mapper, joint))
            capture_inputs.append(joint)
            mapper_modules[layer, target] = mapper
            joint_modules[layer, target] = joint

    storage: dict[str, list[np.ndarray]] = {}
    sample_ids: list[str] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        requests = [generation_request(example) for example in chunk]
        with ActivationCapture(
            model,
            module_names,
            capture_inputs=capture_inputs,
            to_cpu=False,
        ) as captured:
            expanded = generator.prepare_many(requests)
            output = model.language_model(
                inputs_embeds=expanded.inputs_embeds,
                attention_mask=expanded.attention_mask,
                position_ids=expanded.position_ids,
                fusion_state=(expanded.fusion_state if fusion_layers else None),
                output_hidden_states=True,
            )
        sample_ids.extend(example.id for example in chunk)
        native = captured.values["vision_encoder"]
        if len(native) != 1:
            raise RuntimeError("vision encoder capture did not produce exactly one batch")
        _append(storage, "vision.selected", native[0].mean(dim=1))
        if "multimodal_projector" in captured.values:
            projected = captured.values["multimodal_projector"]
            if len(projected) != 1:
                raise RuntimeError("projector capture did not produce exactly one batch")
            _append(storage, "vision.projected", projected[0].mean(dim=1))
        for layer in fusion_layers:
            fusion = model.language_model.model.layers[layer].self_attn.fusion
            assert fusion is not None
            for target in fusion.targets:
                mapped = captured.values[mapper_modules[layer, target]]
                joint = captured.values[joint_modules[layer, target]]
                if len(mapped) != 1 or len(joint) != 1:
                    raise RuntimeError(
                        f"fusion layer {layer}, target {target} executed an unexpected number "
                        "of times"
                    )
                _append(
                    storage,
                    f"fusion.{layer}.{target}.mapped_text",
                    _masked_mean(mapped[0], expanded.fusion_state.text_mask),
                )
                _append(
                    storage,
                    f"fusion.{layer}.{target}.joint",
                    _masked_mean(joint[0], expanded.attention_mask),
                )
        if output.hidden_states is None:
            raise RuntimeError("language model did not return requested hidden states")
        for layer, hidden in enumerate(output.hidden_states):
            _append(storage, f"hidden.{layer}", _masked_mean(hidden, expanded.attention_mask))
            _append(storage, f"hidden.last.{layer}", _last_valid(hidden, expanded.attention_mask))
    arrays = {key: np.concatenate(parts, axis=0) for key, parts in storage.items()}
    if any(value.shape[0] != len(sample_ids) for value in arrays.values()):
        raise RuntimeError("captured arrays lost sample alignment")
    return tuple(sample_ids), arrays


def _text_only_prompt(prompt: str) -> str:
    """Remove image placeholders while preserving the benchmark conversation text."""

    return prompt.replace("<image>\n", "").replace("<image>", "").strip()


@torch.inference_mode()
def capture_huggingface_representations(
    *,
    kind: str,
    checkpoint: str,
    revision: str,
    examples: Sequence[EvaluationExample],
    batch_size: int,
    device: str,
    dtype: str,
    attention_backend: str | None = None,
    image_aspect_ratio: str = "pad",
    local_files_only: bool = False,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Capture a common terminal-token geometry from frozen HF references.

    ``hidden.last.<layer>`` is the common comparison contract across Vicuna,
    LLaVA, and Inverse-LLaVA. It represents the final valid prompt token after
    every transformer layer. The LLaVA path additionally records the mean raw
    and projected visual patch features observed at its multimodal projector.
    """

    if kind not in {"hf-causal", "hf-llava"}:
        raise ValueError(f"unsupported Hugging Face representation kind: {kind}")
    if revision in {"", "main", "pending-freeze"}:
        raise ValueError("representation capture requires an immutable model revision")
    if batch_size <= 0 or not examples:
        raise ValueError("representation capture requires examples and a positive batch size")
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    storage: dict[str, list[np.ndarray]] = {}
    sample_ids: list[str] = []

    if kind == "hf-causal":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            use_fast=False,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
            "local_files_only": local_files_only,
        }
        if attention_backend is not None:
            model_kwargs["attn_implementation"] = attention_backend
        model = AutoModelForCausalLM.from_pretrained(checkpoint, **model_kwargs).to(device)
        model.eval()
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            encoded = tokenizer(
                [_text_only_prompt(example.prompt) for example in chunk],
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(
                **encoded,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None:
                raise RuntimeError("reference causal model returned no hidden states")
            attention_mask = encoded["attention_mask"].bool()
            for layer, hidden in enumerate(output.hidden_states):
                _append(storage, f"hidden.last.{layer}", _last_valid(hidden, attention_mask))
            sample_ids.extend(example.id for example in chunk)
    else:
        if batch_size != 1:
            raise ValueError("hf-llava representation capture uses batch size 1 for exact prompts")
        if any(len(example.images) != 1 for example in examples):
            raise ValueError("hf-llava representation capture requires one image per sample")
        from PIL import Image

        from invllava.data.collate import expand_to_square, image_mean_background
        from invllava.runtime.huggingface import load_hf_llava_runtime

        runtime = load_hf_llava_runtime(
            checkpoint,
            revision=revision,
            dtype=torch_dtype,
            device=device,
            attention_backend=attention_backend,
            local_files_only=local_files_only,
        )
        processor = runtime.processor
        model = runtime.model
        projectors = [
            (name, module)
            for name, module in model.named_modules()
            if name.endswith("multi_modal_projector")
        ]
        if len(projectors) != 1:
            raise RuntimeError(
                "expected one Hugging Face LLaVA multimodal projector, found "
                f"{[name for name, _ in projectors]}"
            )
        projector_inputs: list[Tensor] = []
        projector_outputs: list[Tensor] = []

        def capture_projector(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not inputs or not isinstance(inputs[0], Tensor) or not isinstance(output, Tensor):
                raise TypeError("LLaVA projector input and output must be tensors")
            projector_inputs.append(inputs[0].detach())
            projector_outputs.append(output.detach())

        handle = projectors[0][1].register_forward_hook(capture_projector)
        try:
            for example in examples:
                projector_inputs.clear()
                projector_outputs.clear()
                with Image.open(example.images[0]) as source:
                    image = source.convert("RGB")
                    if image_aspect_ratio == "pad":
                        image = expand_to_square(
                            image,
                            image_mean_background(processor.image_processor.image_mean),
                        )
                    elif image_aspect_ratio != "square":
                        raise ValueError(
                            f"unsupported reference image aspect ratio: {image_aspect_ratio}"
                        )
                    encoded = processor(
                        text=example.prompt,
                        images=image,
                        return_tensors="pt",
                    )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                output = model(
                    **encoded,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if output.hidden_states is None:
                    raise RuntimeError("reference LLaVA model returned no hidden states")
                if len(projector_inputs) != 1 or len(projector_outputs) != 1:
                    raise RuntimeError("LLaVA projector did not execute exactly once")
                _append(storage, "vision.selected", projector_inputs[0].mean(dim=1))
                _append(storage, "vision.projected", projector_outputs[0].mean(dim=1))
                for layer, hidden in enumerate(output.hidden_states):
                    _append(storage, f"hidden.last.{layer}", hidden[:, -1])
                sample_ids.append(example.id)
        finally:
            handle.remove()

    arrays = {key: np.concatenate(parts, axis=0) for key, parts in storage.items()}
    if any(value.shape[0] != len(sample_ids) for value in arrays.values()):
        raise RuntimeError("captured reference arrays lost sample alignment")
    return tuple(sample_ids), arrays


def write_representation_artifact(
    destination: str | Path,
    metadata_path: str | Path,
    *,
    sample_ids: Sequence[str],
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    pooling: str = "sample mean over visual patches or valid sequence positions",
) -> None:
    destination = Path(destination)
    metadata_path = Path(metadata_path)
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("representation artifacts are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".npz", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                sample_ids=np.asarray(sample_ids, dtype=str),
                **arrays,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    atomic_write_json(
        metadata_path,
        {
            **metadata,
            "format": "invllava-representations-v1",
            "pooling": pooling,
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "arrays": {key: list(value.shape) for key, value in sorted(arrays.items())},
            "artifact_sha256": sha256_file(destination),
        },
    )
