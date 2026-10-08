#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This bootstrap recipe targets x86_64. ARM64 uses the separately qualified NVIDIA runtime and harness overlay." >&2
  exit 2
fi
if [[ ! -f uv.lock ]]; then
  echo "uv.lock is missing. Resolve and review it before evaluation." >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --disable-pip-version-check "uv==0.12.7"
fi
if [[ ! -f .venv-language/pyvenv.cfg ]]; then
  uv venv --system-site-packages .venv-language
elif ! python -c 'from pathlib import Path; assert "include-system-site-packages = true" in Path(".venv-language/pyvenv.cfg").read_text()'; then
  echo "Existing .venv-language does not inherit NGC PyTorch; refusing to reuse it." >&2
  exit 2
fi
# shellcheck source=/dev/null
. .venv-language/bin/activate
uv sync --active --frozen --link-mode copy --no-dev --extra language-eval --no-install-package torch
python -c 'from importlib.metadata import version; import lm_eval; print("language evaluation environment ready", version("lm-eval"))'
