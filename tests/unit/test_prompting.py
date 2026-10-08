from dataclasses import dataclass

import pytest

from invllava.prompting import (
    VICUNA_V1_SYSTEM,
    format_vicuna_v1,
    format_vicuna_v1_user_prompt,
)


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


def test_vicuna_training_and_inference_prefix_are_identical() -> None:
    user = "<image>\nWhat is shown?"
    prefix = format_vicuna_v1_user_prompt(user)
    assert prefix == f"{VICUNA_V1_SYSTEM} USER: {user} ASSISTANT:"
    rendered = format_vicuna_v1((Turn("user", user), Turn("assistant", "A cat.")))
    assert rendered == prefix + " A cat.</s>"


def test_vicuna_formatter_rejects_role_order_errors() -> None:
    with pytest.raises(ValueError, match="expected user"):
        format_vicuna_v1((Turn("assistant", "wrong"),))
