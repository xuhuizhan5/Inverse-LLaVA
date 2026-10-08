from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from invllava.artifacts.source import (
    DEFAULT_INPUTS,
    checkout_source_sha256,
    execution_source_sha256,
)


def test_execution_source_includes_dependency_lock() -> None:
    assert "uv.lock" in DEFAULT_INPUTS
    assert "requirements.txt" in DEFAULT_INPUTS
    assert ".env" not in DEFAULT_INPUTS


def _checkout(root: Path) -> Path:
    for relative in DEFAULT_INPUTS:
        target = root / relative
        if "." in target.name or target.name in {"LICENSE", "NOTICE"}:
            target.write_text("fixture\n")
        else:
            target.mkdir(parents=True)
    package = root / "src/invllava"
    package.mkdir()
    (package / "cli.py").write_text("# fixture\n")
    return root


def test_checkout_source_is_captured_without_git_or_environment(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    expected = execution_source_sha256(root)[0]
    assert checkout_source_sha256(root) == expected
    assert checkout_source_sha256(root, expected=expected) == expected
    (root / ".env").write_text("PRIVATE_FIXTURE=excluded\n")
    assert checkout_source_sha256(root) == expected


def test_changed_checkout_rejects_a_stale_source_stamp(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    expected = checkout_source_sha256(root)
    (root / "src/invllava/cli.py").write_text("# changed fixture\n")
    with pytest.raises(ValueError, match="differs from the imported source checkout"):
        checkout_source_sha256(root, expected=expected)


def test_incomplete_checkout_fails_instead_of_accepting_a_stamp(tmp_path: Path) -> None:
    root = _checkout(tmp_path)
    (root / "uv.lock").unlink()
    with pytest.raises(FileNotFoundError):
        checkout_source_sha256(root, expected="a" * 64)


def test_installed_package_does_not_claim_the_callers_working_tree(tmp_path: Path) -> None:
    assert checkout_source_sha256(tmp_path) is None
    assert checkout_source_sha256(tmp_path, expected="a" * 64) == "a" * 64


def test_predict_rejects_stale_source_before_model_arguments(monkeypatch) -> None:
    from argparse import Namespace

    from invllava.cli import command_predict

    monkeypatch.setenv("INVLLAVA_EXECUTION_SOURCE_SHA256", "0" * 64)
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "1")
    # No benchmark/backend/model arguments are supplied. Reaching their use
    # would fail this test, so the source gate must execute first.
    with pytest.raises(ValueError, match="differs from the imported source checkout"):
        command_predict(Namespace(allow_download=False))


def test_source_hash_script_runs_without_pythonpath() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/hash_execution_source.py"),
            str(repository_root),
            "--details",
        ],
        cwd="/tmp",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert re.fullmatch(
        r"sha256=[0-9a-f]{64} files=[1-9][0-9]* bytes=[1-9][0-9]*\n",
        result.stdout,
    )
