"""Content identity for files that can change scientific execution."""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_INPUTS = (
    "src",
    "configs",
    "scripts",
    "tests",
    "requirements",
    "containers",
    "tools",
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
    "third_party/licenses",
)


def _is_platform_metadata(path: Path) -> bool:
    return path.name == ".DS_Store" or path.name.startswith("._") or "__MACOSX" in path.parts


def execution_source_files(root: Path) -> tuple[Path, ...]:
    """Return the ordered, secret-free files staged for scientific execution."""

    root = root.resolve()
    files: list[Path] = []
    for relative in DEFAULT_INPUTS:
        candidate = root / relative
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
        else:
            raise FileNotFoundError(candidate)

    ordered = tuple(sorted(files, key=lambda item: item.relative_to(root).as_posix()))
    metadata_files = [path.relative_to(root) for path in ordered if _is_platform_metadata(path)]
    if metadata_files:
        preview = ", ".join(path.as_posix() for path in metadata_files[:10])
        raise ValueError(f"execution source contains platform metadata files: {preview}")
    for path in ordered:
        if path.is_symlink():
            raise ValueError(f"execution source contains a symlink: {path.relative_to(root)}")
    return ordered


def execution_source_sha256(root: Path) -> tuple[str, int, int]:
    root = root.resolve()
    files = execution_source_files(root)

    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        total_bytes += len(payload)
    return digest.hexdigest(), len(files), total_bytes


def checkout_source_sha256(root: Path, *, expected: str | None = None) -> str | None:
    """Identify a source checkout and reject a stale externally supplied digest.

    Callers pass the imported package's root, rather than an arbitrary working
    directory. A wheel lacks the checkout inputs and retains explicit provenance
    supplied by its deployment. An incomplete checkout fails the normal audit.
    """

    if not (root / "src/invllava/cli.py").is_file():
        return expected
    actual, _, _ = execution_source_sha256(root)
    if expected is not None and expected != actual:
        raise ValueError("execution-source identity differs from the imported source checkout")
    return actual
