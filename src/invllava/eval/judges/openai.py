from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JudgeRequest:
    sample_id: str
    model: str
    system: str
    prompt: str
    temperature: float = 0.0

    @property
    def request_id(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:20]


def execute_request(request: JudgeRequest) -> dict[str, Any]:
    """Execute only after model and prompt are frozen in the benchmark card."""

    from openai import OpenAI

    response = OpenAI().responses.create(
        model=request.model,
        temperature=request.temperature,
        input=[
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt},
        ],
    )
    return {
        "request_id": request.request_id,
        "sample_id": request.sample_id,
        "model": request.model,
        "response_id": response.id,
        "output_text": response.output_text,
    }
