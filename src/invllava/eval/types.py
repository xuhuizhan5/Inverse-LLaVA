from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class EvaluationExample:
    id: str
    prompt: str
    images: tuple[Path, ...]
    references: tuple[str, ...] = ()
    choices: tuple[str, ...] = ()
    group_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    id: str
    prompt: str
    images: tuple[Path, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def generation_request(example: EvaluationExample) -> GenerationRequest:
    forbidden = ("answer", "reference", "ground_truth", "label")
    safe_metadata = {
        key: value
        for key, value in example.metadata.items()
        if not any(fragment in key.lower() for fragment in forbidden)
    }
    return GenerationRequest(example.id, example.prompt, example.images, safe_metadata)


@dataclass(frozen=True)
class Score:
    value: float
    count: int
    details: dict[str, Any] = field(default_factory=dict)


class ScoringProtocol(Protocol):
    id: str

    def score(self, predictions: dict[str, str], examples: list[EvaluationExample]) -> Score: ...
