from types import SimpleNamespace

import torch
from torch import nn

from invllava.data.collate import (
    IGNORE_INDEX,
    IMAGE_TOKEN_ID,
    MultimodalCollator,
    encode_vicuna_v1,
    tokenize_with_image_placeholder,
)
from invllava.data.types import ConversationSample, Turn
from invllava.model.modeling import InverseLLaVAForConditionalGeneration


class _CharacterTokenizer:
    bos_token_id = 1

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        verbose: bool = True,
    ) -> SimpleNamespace:
        assert verbose is False
        ids = ([self.bos_token_id] if add_special_tokens else []) + [10 + ord(c) for c in text]
        return SimpleNamespace(input_ids=ids)


class _VicunaBoundaryTokenizer:
    """Tiny tokenizer with one-token BOS/EOS boundaries like SentencePiece."""

    bos_token_id = 1
    pad_token_id = 0
    model_max_length = 4096

    def __init__(self, *, extra_later_round_boundary: bool = False) -> None:
        self.extra_later_round_boundary = extra_later_round_boundary

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        verbose: bool = True,
    ) -> SimpleNamespace:
        assert verbose is False
        ids = [self.bos_token_id] if add_special_tokens else []
        if self.extra_later_round_boundary and text.startswith(" USER"):
            ids.append(3)
        cursor = 0
        while cursor < len(text):
            if text.startswith("</s>", cursor):
                ids.append(2)
                cursor += 4
            else:
                ids.append(10 + ord(text[cursor]))
                cursor += 1
        return SimpleNamespace(input_ids=ids)


def test_image_placeholder_is_an_out_of_vocabulary_structural_sentinel() -> None:
    tokenizer = _CharacterTokenizer()
    ids = tokenize_with_image_placeholder(tokenizer, "left<image>middle<image>right")
    assert ids.count(IMAGE_TOKEN_ID) == 2
    assert ids.count(tokenizer.bos_token_id) == 1


def test_image_placeholder_can_omit_special_tokens_without_changing_sentinels() -> None:
    tokenizer = _CharacterTokenizer()
    ids = tokenize_with_image_placeholder(
        tokenizer,
        "left<image>right",
        add_special_tokens=False,
    )

    assert ids.count(IMAGE_TOKEN_ID) == 1
    assert tokenizer.bos_token_id not in ids


def test_vicuna_masking_uses_unadjusted_round_boundaries_when_they_match() -> None:
    tokenizer = _VicunaBoundaryTokenizer()
    turns = (
        Turn(role="user", text="<image> first"),
        Turn(role="assistant", text="one"),
        Turn(role="user", text="second"),
        Turn(role="assistant", text="two"),
    )

    input_ids, labels = encode_vicuna_v1(tokenizer, turns)

    assert len(input_ids) == len(labels)
    assert input_ids.count(IMAGE_TOKEN_ID) == 1
    assert labels[0] == IGNORE_INDEX
    assert any(label != IGNORE_INDEX for label in labels)


def test_vicuna_masking_supports_historical_later_round_adjustment() -> None:
    tokenizer = _VicunaBoundaryTokenizer(extra_later_round_boundary=True)
    turns = (
        Turn(role="user", text="first"),
        Turn(role="assistant", text="one"),
        Turn(role="user", text="second"),
        Turn(role="assistant", text="two"),
    )

    input_ids, labels = encode_vicuna_v1(tokenizer, turns)

    assert len(input_ids) == len(labels)
    assert labels[0] == IGNORE_INDEX
    assert any(label != IGNORE_INDEX for label in labels)


def test_vicuna_masking_matches_llava_for_ambiguous_serialized_round() -> None:
    tokenizer = _VicunaBoundaryTokenizer()
    turns = (
        Turn(role="user", text="ASSISTANT: embedded upstream role prefix"),
        Turn(role="assistant", text="answer"),
    )

    input_ids, labels = encode_vicuna_v1(tokenizer, turns)

    assert input_ids
    assert labels == [IGNORE_INDEX] * len(input_ids)


def test_collator_allows_one_legacy_masked_sample_in_supervised_batch() -> None:
    tokenizer = _VicunaBoundaryTokenizer()
    collator = MultimodalCollator(tokenizer, lambda image: torch.empty(0))
    masked = ConversationSample(
        id="masked",
        images=(),
        turns=(
            Turn("user", "ASSISTANT: embedded upstream role prefix"),
            Turn("assistant", "answer"),
        ),
        source="fixture",
    )
    ordinary = ConversationSample(
        id="ordinary",
        images=(),
        turns=(Turn("user", "question"), Turn("assistant", "answer")),
        source="fixture",
    )

    batch = collator((masked, ordinary))

    labels = batch["labels"]
    assert isinstance(labels, torch.Tensor)
    assert bool(labels[0].eq(IGNORE_INDEX).all())
    assert bool(labels[1].ne(IGNORE_INDEX).any())


class _RejectNegativeEmbedding(nn.Embedding):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert bool(input_ids.ge(0).all())
        return super().forward(input_ids)


class _Language(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.model = SimpleNamespace(embed_tokens=_RejectNegativeEmbedding(32, 8))


def test_image_sentinel_never_reaches_the_language_embedding_lookup() -> None:
    model = InverseLLaVAForConditionalGeneration(
        _Language(),  # type: ignore[arg-type]
        nn.Identity(),  # type: ignore[arg-type]
        visual_feature_dim=6,
    )
    expanded = model.prepare(
        input_ids=torch.tensor([[3, IMAGE_TOKEN_ID, 4]]),
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        labels=torch.tensor([[-100, -100, 4]]),
        image_features=[[torch.randn(2, 6)]],
    )
    assert expanded.inputs_embeds.shape == (1, 4, 8)
    assert expanded.fusion_state.vision_mask.tolist() == [[False, True, True, False]]
