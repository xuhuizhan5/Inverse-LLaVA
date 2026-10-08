#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "Usage: $0 <new-output-prefix> -- <command> [arguments...]" >&2
  exit 2
fi
output_prefix="$1"
shift 2
if [[ "$output_prefix" != /* ]]; then
  echo "Nsight output prefix must be absolute." >&2
  exit 2
fi
if [[ -e "${output_prefix}.nsys-rep" || -e "${output_prefix}.sqlite" ]]; then
  echo "Refusing to overwrite an existing Nsight artifact: $output_prefix" >&2
  exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is unavailable; use the PyTorch profiler trace or install NVIDIA Nsight Systems." >&2
  exit 2
fi
mkdir -p "$(dirname "$output_prefix")"
nsys profile \
  --trace=cuda,nvtx,cudnn,cublas,osrt \
  --sample=none \
  --force-overwrite=false \
  --output "$output_prefix" \
  "$@"
