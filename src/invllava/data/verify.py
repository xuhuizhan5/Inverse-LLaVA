from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from invllava.artifacts.hashing import sha256_file


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    size_bytes: int
    sha256: str


def verify_file(path: str | Path, expected_sha256: str | None = None) -> VerifiedFile:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(value)
    digest = sha256_file(value)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError(f"digest mismatch for {value}")
    return VerifiedFile(value.resolve(), value.stat().st_size, digest)
