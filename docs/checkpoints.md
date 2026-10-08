# Checkpoints

Training checkpoints contain trainable safetensors deltas plus the state needed
to resume. A completed, portable model bundle uses six files:

```text
README.md
COMPLETE
checksums.sha256
inverse_llava_config.json
metadata.json
model_delta.safetensors
```

The bundle pins the Vicuna and CLIP revisions and records model configuration,
precision and preprocessing. Backbone weights remain upstream, under their own
licenses. A Hub repository and a local bundle use the same loader; executable
model-repository code is unnecessary.

## Export a completed training checkpoint

```bash
invllava seal-training-checkpoint \
  /workspace/runs/inverse-llava-7b-seed42/checkpoints/step-0005198 \
  --run-dir /workspace/runs/inverse-llava-7b-seed42 \
  --output /workspace/exports/Inverse-LLaVA-7B
```

The destination must be new. Run `invllava verify-run` first, check the exported
hashes, then test fresh loading and a fixed inference panel before distributing
the bundle. Never publish optimizer state or credentials inside a model bundle.

## Load

```python
from invllava.release import load_pretrained

runtime = load_pretrained(
    "/workspace/exports/Inverse-LLaVA-7B",
    cache_dir="/workspace/cache",
    device="cuda",
    dtype="bfloat16",
    lora_execution="unmerged",
)
```

Use `invllava predict --backend release` for the benchmark workflow. Keep its
runtime controls equal to those of the reference evaluation. Merging LoRA is
an optional inference optimization that needs its own numerical check.

Before uploading a model, document its applicable model license separately
from the code license, review its model card, check upstream access requirements,
and verify a fresh-cache download.
The HD and 13B configurations require separate model bundles.
