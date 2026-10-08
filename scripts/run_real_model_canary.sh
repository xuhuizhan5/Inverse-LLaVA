#!/usr/bin/env bash
set -euo pipefail

required=(
  INVLLAVA_REAL_RESULT_ROOT
  INVLLAVA_REAL_FIXTURE_ROOT
  INVLLAVA_PROJECTOR_ROOT
  INVLLAVA_CANARY_RUNTIME_REF
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: $name" >&2
    exit 2
  fi
done
if [[ "${INVLLAVA_ALLOW_DOWNLOADS:-0}" != "1" ]]; then
  echo "Real-model canary requires INVLLAVA_ALLOW_DOWNLOADS=1." >&2
  exit 2
fi

python_bin="${INVLLAVA_CANARY_PYTHON:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Canary Python is not executable: $python_bin" >&2
  exit 2
fi
result_root="$INVLLAVA_REAL_RESULT_ROOT"
fixture_root="$INVLLAVA_REAL_FIXTURE_ROOT/synthetic"
projector_root="$INVLLAVA_PROJECTOR_ROOT"
runtime_ref="$INVLLAVA_CANARY_RUNTIME_REF"
for path in "$result_root" "$fixture_root" "$projector_root"; do
  if [[ -e "$path" ]]; then
    echo "Canary destinations must be new: $path" >&2
    exit 2
  fi
done
mkdir -p "$result_root" "$INVLLAVA_REAL_FIXTURE_ROOT" "$projector_root"

"$python_bin" - <<'PY'
import os

from huggingface_hub import hf_hub_download

print(
    hf_hub_download(
        repo_id="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        filename="mm_projector.bin",
        revision="5414da88308e4287a29f2e9609256458afb0a981",
        local_dir=os.environ["INVLLAVA_PROJECTOR_ROOT"],
    )
)
PY
printf '%s  %s\n' \
  '8a8d5a8fc6030bd16d8ee3df3b20a21c79190a3b0932a798d05f52a1ffdfc215' \
  "$projector_root/mm_projector.bin" | sha256sum -c -
"$python_bin" -m invllava convert-projector \
  "$projector_root/mm_projector.bin" \
  --output "$result_root/converted-projector"
printf '%s  %s\n' \
  '839fa656d8e2243595e396a0b245fe91520abcb3c66255b39b2ba88469869146' \
  "$result_root/converted-projector/projector.safetensors" | sha256sum -c -

"$python_bin" scripts/create_synthetic_fixture.py --output-dir "$fixture_root"

"$python_bin" scripts/launch_train.py configs/experiment/canary_real_model.yaml \
  "$fixture_root/conversations.jsonl" \
  "$fixture_root/conversations.manifest.json" \
  --allow-download \
  --runtime-ref "$runtime_ref" \
  --run-root "$result_root" \
  --run-id inverse-canary

"$python_bin" scripts/launch_train.py configs/experiment/controlled_llava_canary.yaml \
  "$fixture_root/conversations.jsonl" \
  "$fixture_root/conversations.manifest.json" \
  --allow-download \
  --runtime-ref "$runtime_ref" \
  --run-root "$result_root" \
  --run-id controlled-llava-canary \
  --projector-checkpoint "$result_root/converted-projector"

for run in inverse-canary controlled-llava-canary; do
  mkdir -p "$result_root/$run/evaluation"
done
for architecture in inverse controlled-llava; do
  if [[ "$architecture" == "inverse" ]]; then
    experiment=configs/experiment/canary_real_model.yaml
    run=inverse-canary
  else
    experiment=configs/experiment/controlled_llava_canary.yaml
    run=controlled-llava-canary
  fi
  "$python_bin" -m invllava predict configs/benchmark/synthetic_contract.yaml \
    --examples "$fixture_root/evaluation.jsonl" \
    --backend native \
    --experiment "$experiment" \
    --runtime-ref "$runtime_ref" \
    --checkpoint "$result_root/$run/checkpoints/step-0000002" \
    --device cuda --batch-size 1 \
    --output "$result_root/$run/evaluation/predictions.jsonl" \
    --manifest "$result_root/$run/evaluation/manifest.json" \
    --allow-download --allow-unverified
  "$python_bin" -m invllava score configs/benchmark/synthetic_contract.yaml \
    --examples "$fixture_root/evaluation.jsonl" \
    --predictions "$result_root/$run/evaluation/predictions.jsonl" \
    --output "$result_root/$run/evaluation/score.json" \
    --allow-unverified
done

mkdir -p "$result_root/analysis"
"$python_bin" -m invllava capture-representations \
  configs/experiment/canary_real_model.yaml \
  --runtime-ref "$runtime_ref" \
  --checkpoint "$result_root/inverse-canary/checkpoints/step-0000002" \
  --examples "$fixture_root/evaluation.jsonl" \
  --device cuda --batch-size 1 --maximum 2 \
  --output "$result_root/analysis/inverse-representations.npz" \
  --metadata "$result_root/analysis/inverse-representations.json" \
  --allow-download
"$python_bin" -m invllava plot-representation-pca \
  --series vision "$result_root/analysis/inverse-representations.npz" vision.selected \
  --series mapped-text "$result_root/analysis/inverse-representations.npz" \
    fusion.0.q.mapped_text \
  --output "$result_root/analysis/inverse-pca.png" \
  --metadata "$result_root/analysis/inverse-pca.json"
"$python_bin" -m invllava profile-native configs/experiment/canary_real_model.yaml \
  --runtime-ref "$runtime_ref" \
  --checkpoint "$result_root/inverse-canary/checkpoints/step-0000002" \
  --examples "$fixture_root/evaluation.jsonl" \
  --device cuda --batch-sizes 1 --decode-tokens 2 --warmups 3 --repetitions 5 \
  --include-preparation \
  --output "$result_root/analysis/inverse-profile.json" \
  --allow-download

"$python_bin" - <<'PY'
import json
import os
import platform
from pathlib import Path

import torch

from invllava.artifacts.hashing import sha256_file

root = Path(os.environ["INVLLAVA_REAL_RESULT_ROOT"])
paths = [
    root / "converted-projector/projector.safetensors",
    root / "inverse-canary/checkpoint_inventory.json",
    root / "controlled-llava-canary/checkpoint_inventory.json",
    root / "inverse-canary/evaluation/predictions.jsonl",
    root / "controlled-llava-canary/evaluation/predictions.jsonl",
    root / "analysis/inverse-representations.npz",
    root / "analysis/inverse-pca.png",
    root / "analysis/inverse-profile.json",
]
missing = [str(path) for path in paths if not path.is_file()]
if missing:
    raise AssertionError(f"real-model validation is missing artifacts: {missing}")
payload = {
    "schema_version": 1,
    "scope": "real-model-format-and-functional-contract-not-scientific-result",
    "architecture": platform.machine(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "runtime_ref": os.environ["INVLLAVA_CANARY_RUNTIME_REF"],
    "model_revisions": {
        "language": "3321f76e3f527bd14065daf69dad9344000a201d",
        "vision": "ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
        "projector": "5414da88308e4287a29f2e9609256458afb0a981",
    },
    "artifacts": {
        str(path.relative_to(root)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    },
}
(root / "validation-summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
touch "$result_root/REAL_MODEL_CANARY_COMPLETE"
echo "Real-model canary passed: $result_root/validation-summary.json"
