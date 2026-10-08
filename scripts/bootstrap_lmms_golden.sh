#!/usr/bin/env bash
set -euo pipefail

LOCK_FILE="requirements/lmms-eval-lock.txt"
LMMS_ENV="${INVLLAVA_LMMS_ENV:-/workspace/.venv-lmms-golden}"
if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Refusing an unpinned LMMS environment: $LOCK_FILE is missing." >&2
  echo "Resolve requirements/lmms-eval.in with hashes on x86 staging first." >&2
  exit 2
fi
if [[ ! -d "$LMMS_ENV" ]]; then
  uv venv --system-site-packages "$LMMS_ENV"
fi
uv pip sync --python "$LMMS_ENV/bin/python" "$LOCK_FILE"
"$LMMS_ENV/bin/lmms-eval" version
"$LMMS_ENV/bin/lmms-eval" tasks list >/dev/null
echo "LMMS golden-oracle environment passed task discovery: $LMMS_ENV"
