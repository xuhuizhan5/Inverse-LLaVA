#!/usr/bin/env python3
"""Check that local Markdown and HTML links resolve inside the repository."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

MARKDOWN_LINK = re.compile(r"\[[^]]*]\(([^)]+)\)")
HTML_LINK = re.compile(r"(?:href|src)=[\"']([^\"']+)[\"']")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "data:", "#")


def local_targets(text: str) -> list[str]:
    targets = [*MARKDOWN_LINK.findall(text), *HTML_LINK.findall(text)]
    return [target.strip("<>") for target in targets]


def main() -> None:
    root = Path.cwd().resolve()
    missing: list[str] = []
    for document in sorted(root.rglob("*.md")):
        if any(part.startswith(".") for part in document.relative_to(root).parts):
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in local_targets(text):
            if not raw_target or raw_target.startswith(EXTERNAL_PREFIXES):
                continue
            target = unquote(raw_target.split("#", 1)[0])
            candidate = (document.parent / target).resolve()
            if root != candidate and root not in candidate.parents:
                missing.append(
                    f"{document.relative_to(root)}: link escapes repository: {raw_target}"
                )
            elif not candidate.exists():
                missing.append(f"{document.relative_to(root)}: missing {raw_target}")
    if missing:
        raise SystemExit("\n".join(missing))
    print("Local documentation links validated.")


if __name__ == "__main__":
    main()
