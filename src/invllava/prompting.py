"""Conversation formatting shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ConversationTurn(Protocol):
    role: str
    text: str


VICUNA_V1_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
VICUNA_V1_TEMPLATE_ID = "vicuna_v1"


def format_vicuna_v1_user_prompt(user_content: str) -> str:
    """Return the exact prefix used to request one Vicuna-v1 assistant reply."""

    if not user_content.strip():
        raise ValueError("Vicuna user content must not be empty")
    return f"{VICUNA_V1_SYSTEM} USER: {user_content} ASSISTANT:"


def format_vicuna_v1(turns: Sequence[ConversationTurn]) -> str:
    """Render an alternating training conversation with Vicuna-v1 separators."""

    result = VICUNA_V1_SYSTEM
    expected = "user"
    for turn in turns:
        if turn.role != expected:
            raise ValueError(f"expected {expected} turn, got {turn.role}")
        if turn.role == "user":
            result += f" USER: {turn.text} ASSISTANT:"
            expected = "assistant"
        else:
            result += f" {turn.text}</s>"
            expected = "user"
    return result


def vicuna_v1_masking_issue(turns: Sequence[ConversationTurn]) -> str | None:
    """Describe a serialized round that LLaVA-v1 masks in full.

    The reference preprocessing code splits the rendered prompt on a literal
    assistant delimiter. Upstream text can itself begin with that delimiter;
    LLaVA then keeps the input tokens and masks every target token for the
    affected sample. Detect that compatibility case before model allocation so
    it is explicit in the run artifacts.
    """

    prompt = format_vicuna_v1(turns)
    separator = " ASSISTANT: "
    for round_index, conversation_round in enumerate(prompt.split("</s>")):
        if not conversation_round:
            break
        part_count = len(conversation_round.split(separator))
        if part_count != 2:
            return f"round {round_index} contains {part_count - 1} serialized assistant delimiters"
    return None
