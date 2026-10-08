"""Native Transformers inference for contemporary single-image references.

Benchmark text remains fixed. Only the declared Vicuna conversation wrapper is
replaced by the reference processor's own chat template. This adapter does not
load research-model weights or redefine any benchmark's scoring rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import torch
from PIL import Image

from invllava.eval.types import GenerationRequest
from invllava.prompting import VICUNA_V1_SYSTEM

PROMPT_ADAPTER = "vicuna-v1-single-user-to-native-v1"


def user_content(prompt: str) -> str:
    """Remove the exact known wrapper, rejecting ambiguous/multiturn inputs."""

    prefix, suffix = f"{VICUNA_V1_SYSTEM} USER: ", " ASSISTANT:"
    if not prompt.startswith(prefix) or not prompt.endswith(suffix):
        raise ValueError("native reference expects the declared single-turn Vicuna-v1 wrapper")
    content = prompt[len(prefix) : -len(suffix)]
    if any(marker in content for marker in (" USER:", " ASSISTANT:", "</s>")):
        raise ValueError("ambiguous or multiturn reference prompt")
    if content.count("<image>") != 1 or not content.replace("<image>", "").strip():
        raise ValueError("reference prompt must have one image placeholder and nonempty text")
    return content


def native_messages(request: GenerationRequest, image: Image.Image) -> list[dict[str, Any]]:
    """Preserve user text and image position without consulting answer metadata."""

    if len(request.images) != 1:
        raise ValueError("native multimodal reference requires exactly one image per example")
    before, after = user_content(request.prompt).split("<image>")
    content: list[dict[str, Any]] = []
    if before:
        content.append({"type": "text", "text": before})
    content.append({"type": "image", "image": image})
    if after:
        content.append({"type": "text", "text": after})
    return [{"role": "user", "content": content}]


class HuggingFaceMultimodalGenerator:
    """Use AutoProcessor/AutoModelForImageTextToText with native image sizing."""

    def __init__(
        self,
        checkpoint: str,
        *,
        revision: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_new_tokens: int = 128,
        attention_backend: str = "sdpa",
        local_files_only: bool = False,
    ) -> None:
        if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("native reference requires a full immutable Hub commit SHA")
        if max_new_tokens <= 0 or attention_backend not in {"sdpa", "eager"}:
            raise ValueError("invalid reference generation/attention configuration")
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            checkpoint,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        if not self.processor.chat_template:
            raise ValueError("reference processor has no native chat template")
        self.processor.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                checkpoint,
                revision=revision,
                dtype=dtype,
                attn_implementation=attention_backend,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
            .to(device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.max_new_tokens = max_new_tokens
        configuration = {
            "chat_template": self.processor.chat_template,
            "image_processor": self.processor.image_processor.to_dict(),
        }
        self.provenance = {
            "prompt_adapter": PROMPT_ADAPTER,
            "processor_configuration_sha256": hashlib.sha256(
                json.dumps(configuration, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "image_policy": "native-processor-defaults",
            "transformers_version": transformers.__version__,
        }

    @torch.inference_mode()
    def generate_many(self, requests: Sequence[GenerationRequest]) -> list[str]:
        if not requests:
            raise ValueError("reference generation requires at least one example")
        with ExitStack() as stack:
            conversations = []
            for request in requests:
                if len(request.images) != 1:
                    raise ValueError("native reference requires one image per example")
                original = stack.enter_context(Image.open(request.images[0]))
                rgb = original.convert("RGB")
                stack.callback(rgb.close)
                conversations.append(native_messages(request, rgb))
            inputs = self.processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            width = inputs["input_ids"].shape[1]
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
            if generated.shape[0] != len(requests) or generated.shape[1] < width:
                raise RuntimeError("unexpected generated sequence shape")
            if not torch.equal(generated[:, :width], inputs["input_ids"]):
                raise RuntimeError("reference generate did not preserve its decoder input prefix")
            return self.processor.batch_decode(
                generated[:, width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
