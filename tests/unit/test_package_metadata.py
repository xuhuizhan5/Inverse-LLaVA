import re
from pathlib import Path

import invllava


def test_package_version_matches_project_metadata() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, flags=re.MULTILINE)
    assert match is not None
    assert invllava.__version__ == match.group(1)
