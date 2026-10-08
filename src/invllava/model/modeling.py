from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from torch import Tensor, nn

from invllava.model.llama.modeling import InverseLlamaForCausalLM
from invllava.model.sequence import expand_multimodal_sequence
from invllava.model.types import ExpandedSequence, LanguageModelOutput
from invllava.model.vision import VisionEncoder


def _with_training_token_counts(
    output: LanguageModelOutput, expanded: ExpandedSequence
) -> LanguageModelOutput:
    """Attach exposure from the same labels used by the loss, without changing it."""
    if expanded.labels is None:
        return output
    return replace(
        output,
        supervised_token_count=expanded.labels[:, 1:].ne(-100).sum(),
        expanded_token_count=expanded.attention_mask.sum(),
    )


def _zero_parameter_dependency(module: nn.Module, reference: Tensor) -> Tensor:
    """Connect trainable parameters to a numerically unchanged DDP graph.

    A modality-grouped text-only batch legitimately bypasses the early visual
    projector. Touching one element of each trainable tensor with coefficient
    zero gives DDP a zero gradient instead of making every step pay for
    ``find_unused_parameters=True``.
    """

    dependency = reference.new_zeros(())
    for parameter in module.parameters():
        if parameter.requires_grad and parameter.numel():
            dependency = dependency + parameter.reshape(-1)[0] * 0.0
    return dependency


class InverseLLaVAForConditionalGeneration(nn.Module):
    """Composition root; vision and language never communicate through globals."""

    def __init__(
        self,
        language_model: InverseLlamaForCausalLM,
        vision_encoder: VisionEncoder,
        *,
        image_token_id: int = -200,
        visual_feature_dim: int,
        max_length: int = 2048,
        padding_side: str = "right",
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.image_token_id = image_token_id
        self.visual_feature_dim = visual_feature_dim
        self.max_length = max_length
        self.padding_side = padding_side

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        return self.vision_encoder(pixel_values)

    def prepare(
        self,
        *,
        input_ids: Tensor,
        image_features: Sequence[Tensor | Sequence[Tensor]],
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> ExpandedSequence:
        safe_ids = input_ids.masked_fill(
            input_ids.eq(self.image_token_id), self.language_model.config.pad_token_id
        )
        embeddings = self.language_model.model.embed_tokens(safe_ids)
        return expand_multimodal_sequence(
            input_ids=input_ids,
            token_embeddings=embeddings,
            image_features=image_features,
            image_token_id=self.image_token_id,
            attention_mask=attention_mask,
            labels=labels,
            max_length=self.max_length,
            visual_feature_dim=self.visual_feature_dim,
            padding_side=self.padding_side,
        )

    def forward(
        self,
        *,
        input_ids: Tensor,
        image_features: Sequence[Tensor | Sequence[Tensor]],
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> LanguageModelOutput:
        expanded = self.prepare(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        output = self.language_model(
            inputs_embeds=expanded.inputs_embeds,
            attention_mask=expanded.attention_mask,
            position_ids=expanded.position_ids,
            labels=expanded.labels,
            fusion_state=expanded.fusion_state,
            output_hidden_states=output_hidden_states,
        )
        return _with_training_token_counts(output, expanded)


class LLaVAReferenceForConditionalGeneration(nn.Module):
    """Controlled LLaVA-1.5 early-projector baseline on the same Llama core."""

    def __init__(
        self,
        language_model: InverseLlamaForCausalLM,
        vision_encoder: VisionEncoder,
        multimodal_projector: nn.Module,
        *,
        image_token_id: int = -200,
        hidden_size: int,
        max_length: int = 2048,
        padding_side: str = "right",
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.multimodal_projector = multimodal_projector
        self.image_token_id = image_token_id
        self.visual_feature_dim = hidden_size
        self.max_length = max_length
        self.padding_side = padding_side

    def encode_images(self, pixel_values: Tensor) -> Tensor:
        return self.multimodal_projector(self.vision_encoder(pixel_values))

    def prepare(
        self,
        *,
        input_ids: Tensor,
        image_features: Sequence[Tensor | Sequence[Tensor]],
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> ExpandedSequence:
        safe_ids = input_ids.masked_fill(
            input_ids.eq(self.image_token_id), self.language_model.config.pad_token_id
        )
        token_embeddings = self.language_model.model.embed_tokens(safe_ids)
        expanded = expand_multimodal_sequence(
            input_ids=input_ids,
            token_embeddings=token_embeddings,
            image_features=image_features,
            image_token_id=self.image_token_id,
            attention_mask=attention_mask,
            labels=labels,
            max_length=self.max_length,
            visual_feature_dim=self.visual_feature_dim,
            padding_side=self.padding_side,
        )
        # The reference inserts projected patch tokens into the LLM input stream.
        inputs = expanded.inputs_embeds + expanded.fusion_state.visual_features
        if not bool(expanded.fusion_state.vision_mask.any()):
            inputs = inputs + _zero_parameter_dependency(self.multimodal_projector, inputs)
        return ExpandedSequence(
            inputs_embeds=inputs,
            attention_mask=expanded.attention_mask,
            position_ids=expanded.position_ids,
            labels=expanded.labels,
            fusion_state=expanded.fusion_state,
        )

    def forward(
        self,
        *,
        input_ids: Tensor,
        image_features: Sequence[Tensor | Sequence[Tensor]],
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> LanguageModelOutput:
        expanded = self.prepare(
            input_ids=input_ids,
            image_features=image_features,
            attention_mask=attention_mask,
            labels=labels,
        )
        output = self.language_model(
            inputs_embeds=expanded.inputs_embeds,
            attention_mask=expanded.attention_mask,
            position_ids=expanded.position_ids,
            labels=expanded.labels,
            fusion_state=None,
            output_hidden_states=output_hidden_states,
        )
        return _with_training_token_counts(output, expanded)
