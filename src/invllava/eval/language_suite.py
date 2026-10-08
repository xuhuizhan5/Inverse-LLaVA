from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field, model_validator

from invllava.config.schema import StrictModel


class HarnessIdentity(StrictModel):
    package: str
    version: str
    source_commit: str


class LanguageTask(StrictModel):
    id: str
    num_fewshot: int = Field(ge=0)
    primary_metric: str
    filter: str = "none"


class LanguageRetentionSuite(StrictModel):
    id: str
    harness: HarnessIdentity
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    context_length: int = Field(default=2048, ge=2)
    add_bos_token: bool = True
    seed: int = 42
    tasks: tuple[LanguageTask, ...]
    notes: str = ""

    @model_validator(mode="after")
    def validate_suite(self) -> LanguageRetentionSuite:
        if self.harness.package != "lm-eval":
            raise ValueError("the implemented language suite boundary is lm-eval")
        if len(self.harness.source_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.harness.source_commit.lower()
        ):
            raise ValueError("harness.source_commit must be a full Git commit")
        task_ids = [task.id for task in self.tasks]
        if not task_ids or len(task_ids) != len(set(task_ids)):
            raise ValueError("language tasks must be non-empty and unique")
        if self.apply_chat_template or self.fewshot_as_multiturn:
            raise ValueError("language retention uses the raw causal-LM protocol")
        return self


def load_language_suite(path: str | Path) -> LanguageRetentionSuite:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return LanguageRetentionSuite.model_validate(payload)
