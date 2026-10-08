from __future__ import annotations

import subprocess
import tarfile
from pathlib import Path


def test_source_archive_uses_execution_allowlist(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    destination = tmp_path / "source.tar.gz"

    subprocess.run(
        [
            "bash",
            "tools/artifacts/create_source_archive.sh",
            str(repository_root),
            str(destination),
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with tarfile.open(destination, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
    assert "pyproject.toml" in names
    assert "requirements.txt" in names
    assert "src/invllava/__init__.py" in names
    assert {
        "LICENSE",
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "third_party/licenses/Pythia-BSD.txt",
    } <= names
    assert ".env" not in names
    assert all(not name.startswith(("secrets/", "credentials/", ".runpod/")) for name in names)
    assert all(member.uid == 0 and member.gid == 0 for member in members)
