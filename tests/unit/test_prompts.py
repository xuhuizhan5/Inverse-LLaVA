from pathlib import Path

import pytest
import yaml

from invllava.config.schema import BenchmarkSpec
from invllava.eval.prompts import format_choices, render_benchmark_prompt, render_prompt
from invllava.prompting import format_vicuna_v1_user_prompt


def test_render_prompt_is_strict_and_exact() -> None:
    assert render_prompt("<image>\n{question}", {"question": "What?"}) == "<image>\nWhat?"
    with pytest.raises(ValueError, match="missing values"):
        render_prompt("{question}\n{choices}", {"question": "What?"})
    with pytest.raises(ValueError, match="invalid field"):
        render_prompt("{row[question]}", {"row": {"question": "What?"}})


def test_format_choices_has_a_b_labels() -> None:
    assert format_choices(("red", "blue")) == "A. red\nB. blue"
    with pytest.raises(ValueError, match="empty"):
        format_choices(())


def test_benchmark_prompt_applies_declared_conversation_template() -> None:
    spec = BenchmarkSpec.model_validate(
        yaml.safe_load(Path("configs/benchmark/gqa.yaml").read_text(encoding="utf-8"))
    )
    assert render_benchmark_prompt(spec, {"question": "Where?"}) == (
        format_vicuna_v1_user_prompt(
            "<image>\nWhere?\nAnswer the question using a single word or phrase."
        )
    )
