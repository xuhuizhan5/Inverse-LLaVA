from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    role: str
    text: str


@dataclass(frozen=True)
class ConversationSample:
    id: str
    images: tuple[Path, ...]
    turns: tuple[Turn, ...]
    source: str

    def validate(self, image_token: str = "<image>") -> None:
        placeholders = sum(turn.text.count(image_token) for turn in self.turns)
        if placeholders != len(self.images):
            raise ValueError(
                f"sample {self.id}: {placeholders} image placeholders but {len(self.images)} images"
            )
        if not self.turns:
            raise ValueError(f"sample {self.id}: conversation is empty")
        expected = "user"
        for index, turn in enumerate(self.turns):
            if turn.role != expected:
                raise ValueError(
                    f"sample {self.id}: turn {index} must be {expected}, got {turn.role}"
                )
            expected = "assistant" if expected == "user" else "user"
        if self.turns[-1].role != "assistant":
            raise ValueError(f"sample {self.id}: conversation must end with an assistant turn")
