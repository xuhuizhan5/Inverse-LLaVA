"""Provider-neutral cache routing for model and compiler dependencies."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_cache(cache_root: str | Path) -> None:
    """Route library caches under one execution-owned root.

    Explicit environment variables take precedence because container launchers
    may mount each cache independently. Otherwise all heavy caches inherit the
    runtime configuration's root instead of writing into a user's home folder.
    """

    root = Path(cache_root)
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(root / "torch_extensions"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(root / "triton"))
