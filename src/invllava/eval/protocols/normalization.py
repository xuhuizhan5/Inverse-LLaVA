from __future__ import annotations

import re
import string

ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(value: str) -> str:
    value = value.lower().replace("\n", " ").replace("\t", " ").strip()
    value = "".join(character for character in value if character not in string.punctuation)
    value = ARTICLES.sub(" ", value)
    return " ".join(value.split())


def extract_choice(value: str, choices: tuple[str, ...]) -> str | None:
    stripped = value.strip()
    label_match = re.search(r"(?:^|\b)([A-Z])(?:\b|[\).:])", stripped.upper())
    if label_match:
        label = label_match.group(1)
        index = ord(label) - ord("A")
        if 0 <= index < len(choices):
            return label
    normalized = normalize_answer(value)
    matching = [
        index for index, choice in enumerate(choices) if normalize_answer(choice) in normalized
    ]
    if len(matching) == 1:
        return chr(ord("A") + matching[0])
    return None
