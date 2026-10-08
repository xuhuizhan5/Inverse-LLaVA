import pytest

from invllava.artifacts.hashing import optional_sha256_environment


def test_optional_sha256_environment_validates_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVLLAVA_TEST_DIGEST", raising=False)
    assert optional_sha256_environment("INVLLAVA_TEST_DIGEST") is None

    digest = "a" * 64
    monkeypatch.setenv("INVLLAVA_TEST_DIGEST", digest)
    assert optional_sha256_environment("INVLLAVA_TEST_DIGEST") == digest

    monkeypatch.setenv("INVLLAVA_TEST_DIGEST", "A" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        optional_sha256_environment("INVLLAVA_TEST_DIGEST")
