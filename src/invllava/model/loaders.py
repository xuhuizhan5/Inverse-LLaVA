"""Checkpoint loading is strict except for newly introduced fusion parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from invllava.config.schema import FusionSpec, ModelSpec, VisionSpec
from invllava.model.adaptation import apply_lora, set_lora_trainable
from invllava.model.fusion import FusionBlock
from invllava.model.llama.configuration import LlamaArchitecture
from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.modeling import (
    InverseLLaVAForConditionalGeneration,
    LLaVAReferenceForConditionalGeneration,
)
from invllava.model.projector import MLP2xGELUProjector
from invllava.model.vision import VisionEncoder


@dataclass(frozen=True)
class LoadReport:
    source: str
    missing_expected: tuple[str, ...]
    unexpected: tuple[str, ...]
    checkpoint_format: str
    ignored_derived_buffers: tuple[str, ...] = ()
    rotary_frequency_dtype: str | None = None


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _set_fusion_trainability(fusion: FusionBlock, spec: FusionSpec) -> None:
    """Apply the block policy without undoing its component-level exclusions."""

    fusion.requires_grad_(spec.trainable)
    if not spec.mapper_trainable:
        fusion.text_to_vision.requires_grad_(False)
    if spec.scale_mode == "fixed":
        fusion.scales.requires_grad_(False)


def _checkpoint_keys(root: str | Path) -> tuple[set[str], str]:
    root = Path(root)
    for pattern, checkpoint_format in (
        ("*.safetensors.index.json", "safetensors-sharded"),
        ("*.bin.index.json", "pytorch-bin-sharded-weights-only"),
    ):
        indices = sorted(root.glob(pattern))
        if not indices:
            continue
        keys: set[str] = set()
        for path in indices:
            value = json.loads(path.read_text(encoding="utf-8"))
            weight_map = value.get("weight_map")
            if not isinstance(weight_map, dict):
                raise ValueError(f"invalid checkpoint index: {path}")
            keys.update(str(key) for key in weight_map)
        return keys, checkpoint_format
    from safetensors import safe_open

    keys = set()
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            keys.update(stream.keys())
    if keys:
        return keys, "safetensors"
    bin_files = sorted(root.glob("*.bin"))
    if len(bin_files) != 1:
        raise ValueError(f"no unambiguous model checkpoint found under {root}")
    value = torch.load(bin_files[0], map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("single-file PyTorch checkpoint is not a tensor state dictionary")
    return set(value), "pytorch-bin-weights-only"


def _compare_base_keys(
    expected: set[str],
    found: set[str],
    *,
    tied_word_embeddings: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    missing = expected - found
    if tied_word_embeddings and {"model.embed_tokens.weight", "lm_head.weight"}.intersection(found):
        missing -= {"model.embed_tokens.weight", "lm_head.weight"}
    return tuple(sorted(missing)), tuple(sorted(found - expected))


def _validate_rope_contract(value: object, *, rope_theta: float) -> None:
    """Accept no scaling or Transformers' normalized representation of it."""

    if value in (None, {}):
        return
    if not isinstance(value, dict):
        raise ValueError("the native Llama contract received invalid RoPE metadata")
    if set(value) - {"rope_type", "rope_theta"}:
        raise ValueError("the native Llama contract does not implement scaled RoPE")
    if value.get("rope_type") != "default" or float(value.get("rope_theta", rope_theta)) != float(
        rope_theta
    ):
        raise ValueError("the native Llama contract does not implement scaled RoPE")


def build_language_model(
    spec: ModelSpec,
    *,
    attention_backend: str = "sdpa",
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[InverseLlamaForCausalLM, LoadReport]:
    from accelerate import init_empty_weights, load_checkpoint_in_model
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(
        spec.language.checkpoint,
        revision=spec.language.revision,
        trust_remote_code=spec.language.trust_remote_code,
        local_files_only=local_files_only,
        token=token,
    )
    architecture = LlamaArchitecture.from_hf(hf_config)
    _validate_rope_contract(
        getattr(hf_config, "rope_scaling", None), rope_theta=architecture.rope_theta
    )
    if architecture.hidden_size != spec.language.hidden_size:
        raise ValueError("resolved hidden size disagrees with downloaded language configuration")
    if architecture.num_hidden_layers != spec.language.num_layers:
        raise ValueError("resolved layer count disagrees with downloaded language configuration")
    head_dim = architecture.hidden_size // architecture.num_attention_heads
    target_sizes = {
        "q": architecture.num_attention_heads * head_dim,
        "k": architecture.num_key_value_heads * head_dim,
        "v": architecture.num_key_value_heads * head_dim,
    }
    dtype = _torch_dtype(spec.torch_dtype)
    with init_empty_weights():
        model = InverseLlamaForCausalLM(architecture, attention_backend=attention_backend)
    checkpoint_root = snapshot_download(
        spec.language.checkpoint,
        revision=spec.language.revision,
        allow_patterns=(
            "*.safetensors",
            "*.safetensors.index.json",
            "*.bin",
            "*.bin.index.json",
            "config.json",
        ),
        local_files_only=local_files_only,
        token=token,
    )
    checkpoint_keys, checkpoint_format = _checkpoint_keys(checkpoint_root)
    missing_base, unexpected_base = _compare_base_keys(
        set(model.state_dict()),
        checkpoint_keys,
        tied_word_embeddings=architecture.tie_word_embeddings,
    )
    if missing_base or unexpected_base:
        raise RuntimeError(
            "base language checkpoint does not exactly match the native Llama contract; "
            f"missing={missing_base[:8]}, unexpected={unexpected_base[:8]}"
        )
    load_checkpoint_in_model(
        model,
        checkpoint_root,
        device_map={"": "cpu"},
        dtype=dtype,
        offload_state_dict=True,
    )
    # Accelerate's keep_in_fp32_modules applies only to FP16 in the pinned
    # version. Explicitly restore verified unscaled frequencies for BF16 too.
    for layer in model.model.layers:
        layer.self_attn.rotary_emb.set_precision(spec.language.rotary_precision)
    frequency_dtype = torch.float32 if spec.language.rotary_precision == "float32" else dtype
    if any(
        layer.self_attn.rotary_emb.inv_freq.dtype != frequency_dtype for layer in model.model.layers
    ):
        raise RuntimeError("loaded rotary frequency dtype disagrees with the declared policy")
    meta = tuple(name for name, parameter in model.named_parameters() if parameter.is_meta)
    if meta:
        raise RuntimeError(f"base checkpoint left parameters on meta device: {meta[:5]}")

    fusion_layers = {
        index: FusionBlock(
            hidden_size=architecture.hidden_size,
            visual_size=spec.vision.feature_dim,
            target_sizes=target_sizes,
            targets=spec.fusion.targets,
            operator=spec.fusion.operator,
            mapper_rank=spec.fusion.mapper_rank,
            mapper_trainable=spec.fusion.mapper_trainable,
            visual_normalization=spec.fusion.visual_normalization,
            visual_norm_eps=spec.fusion.visual_norm_eps,
            scale_mode=spec.fusion.scale_mode,
            initial_scale=spec.fusion.initial_scale,
            mapper_init_std=spec.fusion.mapper_init_std,
            output_init_std=spec.fusion.output_init_std,
            # Match the submitted launch mechanics: initialize newly created
            # trainable modules with PyTorch's FP32 factory dtype, then cast
            # the complete branch to the configured model/forward dtype.
            parameter_dtype=torch.float32,
        ).to(dtype=dtype)
        for index in spec.fusion.layers
        if spec.fusion.operator != "disabled"
    }
    for index, fusion in fusion_layers.items():
        model.model.layers[index].self_attn.fusion = fusion
    allowed_missing = tuple(
        name
        for name in model.state_dict()
        if any(f"model.layers.{index}.self_attn.fusion." in name for index in fusion_layers)
    )

    if spec.adaptation.method == "lora":
        model.requires_grad_(False)
        apply_lora(
            model,
            target_suffixes=spec.adaptation.target_suffixes,
            rank=spec.adaptation.rank,
            alpha=spec.adaptation.alpha,
            dropout=spec.adaptation.dropout,
            excluded_layer_indices=tuple(
                sorted(
                    set(spec.adaptation.excluded_layer_indices).union(
                        spec.fusion.layers if spec.adaptation.exclude_fusion_layers else ()
                    )
                )
            ),
        )
        if not spec.adaptation.trainable:
            set_lora_trainable(model, False)
    elif spec.adaptation.method == "frozen":
        model.requires_grad_(False)
    for layer in spec.fusion.layers:
        fusion = model.model.layers[layer].self_attn.fusion
        if fusion is not None:
            _set_fusion_trainability(fusion, spec.fusion)
    return model, LoadReport(
        source=spec.language.checkpoint,
        missing_expected=allowed_missing,
        unexpected=tuple(),
        checkpoint_format=checkpoint_format,
        ignored_derived_buffers=tuple(),
        rotary_frequency_dtype=str(frequency_dtype),
    )


def load_image_processor(
    spec: VisionSpec,
    *,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Any:
    """Use one declared image backend in training, evaluation, and model releases."""

    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        spec.checkpoint,
        revision=spec.revision,
        backend=spec.processor_backend,
        local_files_only=local_files_only,
        token=token,
    )
    if getattr(processor, "backend", None) != spec.processor_backend:
        raise RuntimeError("image processor did not honor the declared backend")
    return processor


def build_model(
    spec: ModelSpec,
    *,
    attention_backend: str = "sdpa",
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> tuple[
    InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration,
    LoadReport,
]:
    if (
        spec.architecture == "llava_reference"
        and spec.projector is not None
        and spec.projector.initial_checkpoint_id == "embedded-in-official-checkpoint"
    ):
        raise ValueError(
            "official full-tuned LLaVA checkpoints must use the isolated hf-llava adapter; "
            "native loading is reserved for the controlled Vicuna + projector baseline"
        )
    language, report = build_language_model(
        spec,
        attention_backend=attention_backend,
        local_files_only=local_files_only,
        token=token,
    )
    vision = VisionEncoder.from_pretrained(
        spec.vision.checkpoint,
        revision=spec.vision.revision,
        attention_backend=attention_backend,
        local_files_only=local_files_only,
        token=token,
        feature_layers=spec.vision.feature_layers,
        feature_select=spec.vision.feature_select,
        freeze=spec.vision.freeze,
        dtype=_torch_dtype(spec.torch_dtype),
    )
    if spec.architecture == "inverse_llava":
        model: InverseLLaVAForConditionalGeneration | LLaVAReferenceForConditionalGeneration
        model = InverseLLaVAForConditionalGeneration(
            language,
            vision,
            image_token_id=-200,
            visual_feature_dim=spec.vision.feature_dim,
            max_length=spec.language.max_length,
        )
    else:
        if spec.projector is None:  # Protected by schema validation.
            raise ValueError("LLaVA reference is missing its projector contract")
        projector = MLP2xGELUProjector(
            spec.projector.input_dim,
            spec.projector.output_dim,
        ).to(dtype=_torch_dtype(spec.torch_dtype))
        projector.requires_grad_(spec.projector.trainable)
        model = LLaVAReferenceForConditionalGeneration(
            language,
            vision,
            projector,
            image_token_id=-200,
            hidden_size=spec.language.hidden_size,
            max_length=spec.language.max_length,
        )
    return model, report


def load_tokenizer(
    spec: ModelSpec,
    *,
    local_files_only: bool = False,
    token: str | bool | None = None,
) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.language.checkpoint,
        revision=spec.language.revision,
        use_fast=False,
        trust_remote_code=spec.language.trust_remote_code,
        local_files_only=local_files_only,
        token=token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer
