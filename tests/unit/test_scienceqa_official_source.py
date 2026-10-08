"""The reference audit executes only extraction from its pinned source."""

import pytest

from invllava.artifacts.hashing import sha256_file
from scripts import verify_scienceqa_scores as verifier


def test_changed_official_source_rejected(tmp_path):
    path = tmp_path / "untrusted.py"
    path.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="identity"):
        verifier.official_parser(path)


def test_unexpected_source_structure_rejected(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.py"
    path.write_text("raise RuntimeError('must not execute')\n")
    monkeypatch.setattr(verifier, "OFFICIAL_SHA256", sha256_file(path))
    with pytest.raises(ValueError, match="structure"):
        verifier.official_parser(path)


def test_only_selected_blocks_execute(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.py"
    path.write_text(
        "raise RuntimeError('top-level code must not execute')\n"
        "def get_pred_idx(prediction, choices, options):\n"
        "    return options.index(prediction) if prediction in options[:len(choices)] else -1\n"
        "if pred_text in args.options:\n"
        "    answer = pred_text\n"
        "else:\n"
        "    answer = 'FAILED'\n"
    )
    monkeypatch.setattr(verifier, "OFFICIAL_SHA256", sha256_file(path))
    parse = verifier.official_parser(path)
    assert parse("B", 2) == "B"
    assert parse("C", 2) is None
    assert parse("b", 2) is None
