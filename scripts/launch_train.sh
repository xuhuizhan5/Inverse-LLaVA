#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <experiment.yaml> <prepared.jsonl> <prepared.manifest.json> [launcher options]" >&2
  exit 2
fi
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  exec uv run --no-sync python scripts/launch_train.py "$@"
fi
exec "$python_bin" scripts/launch_train.py "$@"
