#!/usr/bin/env bash
set -euo pipefail

if ! python3 -c 'import ruff' >/dev/null 2>&1; then
  echo 'Install the dev extra before linting: python -m pip install -e ".[dev]"' >&2
  exit 2
fi
if ! command -v shellcheck >/dev/null 2>&1; then
  echo "shellcheck is required for repository linting." >&2
  exit 2
fi

python3 -m ruff check src scripts tests tools
python3 -m ruff format --check src scripts tests tools
find scripts tools -type f -name '*.sh' -exec shellcheck {} +
