#!/usr/bin/env bash
set -euo pipefail

static_cache="$(mktemp -d "${TMPDIR:-/tmp}/invllava-static-pycache.XXXXXX")"
trap 'rm -rf -- "$static_cache"' EXIT
PYTHONPYCACHEPREFIX="$static_cache" python3 -m compileall -q src tests scripts tools
python3 - <<'PY'
import ast
from pathlib import Path

for path in Path("src").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in Path("tests").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in Path("scripts").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
for path in Path("tools").rglob("*.py"):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print("Python syntax validated without importing project dependencies.")
PY
PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs.py
if PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -c 'import pydantic, yaml' >/dev/null 2>&1; then
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/verify_config_catalog.py
else
  echo "Configuration catalog skipped: pydantic and PyYAML are unavailable." >&2
fi
