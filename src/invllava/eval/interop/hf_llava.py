"""Reference-only Hugging Face LLaVA runtime for benchmark adapter goldens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from PIL import Image

from invllava.data.collate import expand_to_square, image_mean_background
from invllava.eval.types import GenerationRequest
from invllava.model.sequence import build_text_only_fusion_state
from invllava.model.types import ExpandedSequence


class HuggingFaceLLaVAGenerator:
    """Run a converted official checkpoint through the standalone HF adapter."""

    def __init__(
        self,
        checkpoint: str,
        *,
        revision: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        max_new_tokens: int = 128,
        attention_backend: str | None = None,
        image_aspect_ratio: Literal["pad", "square"] = "pad",
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> None:
        from invllava.runtime.huggingface import load_hf_llava_runtime

        runtime = load_hf_llava_runtime(
            checkpoint,
            revision=revision,
            dtype=dtype,
            device=device,
            attention_backend=attention_backend,
            local_files_only=local_files_only,
            token=token,
        )
        self.processor = runtime.processor
        self.model = runtime.model
        self.max_new_tokens = max_new_tokens
        self.image_aspect_ratio = image_aspect_ratio
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"

    def _prepare_image(self, image: Image.Image) -> Image.Image:
        rgb = image.convert("RGB")
        if self.image_aspect_ratio == "square":
            return rgb.copy()
        image_processor = getattr(self.processor, "image_processor", None)
        image_mean = getattr(image_processor, "image_mean", None)
        if image_mean is None:
            raise ValueError("padded LLaVA inference requires processor.image_processor.image_mean")
        return expand_to_square(rgb, image_mean_background(image_mean)).copy()

    def _processor_inputs(self, requests: Sequence[GenerationRequest]) -> dict[str, Any]:
        if not requests:
            raise ValueError("Hugging Face LLaVA preparation requires at least one request")
        if any(len(request.images) != 1 for request in requests):
            raise ValueError("batched Hugging Face LLaVA evaluation requires one image per sample")
        images: list[Image.Image] = []
        try:
            for request in requests:
                path = request.images[0]
                with Image.open(path) as image:
                    images.append(self._prepare_image(image))
            inputs: dict[str, Any] = self.processor(
                text=[request.prompt for request in requests],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            return {key: value.to(device) for key, value in inputs.items()}
        finally:
            for image in images:
                image.close()

    def _expanded_from_processor_inputs(self, inputs: dict[str, Any]) -> ExpandedSequence:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].to(dtype=torch.bool)
        multimodal = self.model.model
        inputs_embeds = multimodal.get_input_embeddings()(input_ids)
        image_output = multimodal.get_image_features(
            pixel_values=inputs["pixel_values"],
            image_sizes=inputs.get("image_sizes"),
            return_dict=True,
        )
        image_features = torch.cat(image_output.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask = multimodal.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        return ExpandedSequence(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=None,
            fusion_state=build_text_only_fusion_state(inputs_embeds, feature_dim=1),
        )

    @torch.inference_mode()
    def prepare_many(self, requests: Sequence[GenerationRequest]) -> ExpandedSequence:
        """Prepare the exact merged language sequence for matched profiling."""

        return self._expanded_from_processor_inputs(self._processor_inputs(requests))

    @torch.inference_mode()
    def verify_preparation_parity(
        self,
        requests: Sequence[GenerationRequest],
        *,
        atol: float = 1e-3,
        rtol: float = 1e-3,
    ) -> dict[str, float | bool]:
        """Qualify the expanded-sequence path against the official HF forward route."""

        if atol < 0 or rtol < 0:
            raise ValueError("parity tolerances must be non-negative")
        inputs = self._processor_inputs(requests)
        expanded = self._expanded_from_processor_inputs(inputs)
        shared = {
            "attention_mask": expanded.attention_mask,
            "position_ids": expanded.position_ids,
            "use_cache": True,
            "return_dict": True,
        }
        direct = self.model(**inputs, position_ids=expanded.position_ids, use_cache=True)
        prepared = self.model(inputs_embeds=expanded.inputs_embeds, **shared)
        if direct.logits.shape != prepared.logits.shape:
            raise RuntimeError("HF preparation parity produced different logit shapes")
        last_indices = (
            torch.arange(
                expanded.attention_mask.shape[1],
                device=expanded.attention_mask.device,
            )
            .unsqueeze(0)
            .expand(expanded.attention_mask.shape[0], -1)
            .masked_fill(~expanded.attention_mask, -1)
            .max(dim=1)
            .values
        )
        rows = torch.arange(len(requests), device=last_indices.device)
        direct_last = direct.logits[rows, last_indices]
        prepared_last = prepared.logits[rows, last_indices]
        direct_tokens = direct_last.argmax(dim=-1, keepdim=True)
        prepared_tokens = prepared_last.argmax(dim=-1, keepdim=True)
        attention = torch.cat(
            (
                expanded.attention_mask,
                torch.ones(
                    (len(requests), 1),
                    dtype=torch.bool,
                    device=expanded.attention_mask.device,
                ),
            ),
            dim=1,
        )
        position_ids = attention.long().sum(dim=-1, keepdim=True) - 1
        direct_continuation = self.model(
            input_ids=direct_tokens,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=direct.past_key_values,
            use_cache=True,
            return_dict=True,
        ).logits[:, -1]
        prepared_continuation = self.model(
            input_ids=prepared_tokens,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=prepared.past_key_values,
            use_cache=True,
            return_dict=True,
        ).logits[:, -1]
        prefill_close = torch.allclose(direct_last, prepared_last, atol=atol, rtol=rtol)
        continuation_close = torch.allclose(
            direct_continuation,
            prepared_continuation,
            atol=atol,
            rtol=rtol,
        )
        token_equal = torch.equal(direct_tokens, prepared_tokens)
        report: dict[str, float | bool] = {
            "passed": bool(prefill_close and continuation_close and token_equal),
            "atol": atol,
            "rtol": rtol,
            "prefill_logits_allclose": bool(prefill_close),
            "prefill_logits_max_abs_error": float(
                (direct_last.float() - prepared_last.float()).abs().max().item()
            ),
            "first_token_equal": token_equal,
            "continuation_logits_allclose": bool(continuation_close),
            "continuation_logits_max_abs_error": float(
                (direct_continuation.float() - prepared_continuation.float()).abs().max().item()
            ),
        }
        if not report["passed"]:
            raise RuntimeError(f"HF expanded-sequence preparation parity failed: {report}")
        return report

    @torch.inference_mode()
    def __call__(self, request: GenerationRequest) -> str:
        return self.generate_many((request,))[0]

    @torch.inference_mode()
    def generate_many(self, requests: Sequence[GenerationRequest]) -> list[str]:
        if not requests:
            return []
        inputs = self._processor_inputs(requests)
        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        prompt_length = inputs["input_ids"].shape[1]
        return [
            text.strip()
            for text in self.processor.batch_decode(
                output[:, prompt_length:], skip_special_tokens=True
            )
        ]
