#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-directory>" >&2
  exit 2
fi
output="$1"
if [[ -e "$output" ]]; then
  echo "Environment snapshot destination must be new: $output" >&2
  exit 2
fi
mkdir -p "$output"
if [[ -x .venv/bin/python ]]; then
  snapshot_python=.venv/bin/python
elif [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -x "${VIRTUAL_ENV}/bin/python" ]]; then
  snapshot_python="${VIRTUAL_ENV}/bin/python"
else
  snapshot_python=python
fi
"$snapshot_python" -VV > "$output/python.txt"
"$snapshot_python" -m pip freeze > "$output/pip-freeze.txt"
uname -a > "$output/uname.txt"
git rev-parse HEAD > "$output/git-commit.txt" 2>/dev/null || true
git status --porcelain > "$output/git-status.txt" 2>/dev/null || true
if command -v sha256sum >/dev/null 2>&1; then
  hash_file() { sha256sum "$1"; }
else
  hash_file() { shasum -a 256 "$1"; }
fi
if [[ -f uv.lock ]]; then
  hash_file uv.lock > "$output/uv-lock.sha256"
fi
hash_file pyproject.toml > "$output/pyproject.sha256"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -q > "$output/nvidia-smi.txt"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
    --format=csv > "$output/gpu-inventory.csv"
  nvidia-smi topo -m > "$output/gpu-topology.txt"
fi
