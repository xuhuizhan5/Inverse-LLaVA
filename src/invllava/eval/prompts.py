"""Strict rendering for benchmark prompts declared in protocol YAML.

Prompt text is part of the evaluation protocol.  Converters therefore receive
the template from :class:`BenchmarkSpec` and may only supply named values; they
must not carry a second, hidden copy of the prompt in Python.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

from invllava.config.schema import BenchmarkSpec
from invllava.prompting import format_vicuna_v1_user_prompt


def format_choices(choices: tuple[str, ...]) -> str:
    """Render choices using the A./B./... form used by the frozen protocols."""

    if not choices:
        raise ValueError("cannot render an empty choice list")
    if len(choices) > len(string.ascii_uppercase):
        raise ValueError("choice protocol supports at most 26 options")
    return "\n".join(
        f"{string.ascii_uppercase[index]}. {choice}" for index, choice in enumerate(choices)
    )


def render_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Render a protocol prompt and reject ambiguous template expressions.

    Only simple named fields such as ``{question}`` are accepted.  Attribute
    access, indexing, positional fields, conversions, and format specifiers are
    deliberately rejected so benchmark YAML remains reviewable and data-only.
    """

    if not template:
        raise ValueError("benchmark prompt template must not be empty")
    formatter = string.Formatter()
    fields: set[str] = set()
    for _, field_name, format_spec, conversion in formatter.parse(template):
        if field_name is None:
            continue
        if not field_name or not field_name.isidentifier():
            raise ValueError(f"prompt template has an invalid field: {field_name!r}")
        if format_spec or conversion:
            raise ValueError(f"prompt field {field_name!r} may not use formatting or conversion")
        fields.add(field_name)
    missing = sorted(fields.difference(values))
    if missing:
        raise ValueError("prompt template is missing values for: " + ", ".join(missing))
    rendered = template.format_map(dict(values))
    if not rendered.strip():
        raise ValueError("rendered benchmark prompt is empty")
    return rendered


def render_benchmark_prompt(spec: BenchmarkSpec, values: Mapping[str, Any]) -> str:
    """Render the user template and its declared model conversation wrapper."""

    user_content = render_prompt(spec.prompt_template, values)
    if spec.conversation_template == "vicuna_v1":
        return format_vicuna_v1_user_prompt(user_content)
    raise ValueError(f"unsupported conversation template: {spec.conversation_template}")
