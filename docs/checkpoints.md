# Checkpoints

Training checkpoints contain trainable safetensors deltas plus the state needed
to resume. Exporting produces five runtime files; add a model card and the
applicable model license and notices before publication:

```text
README.md
COMPLETE
checksums.sha256
inverse_llava_config.json
metadata.json
model_delta.safetensors
LICENSE.txt
NOTICE
USE_POLICY.md
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
    max_new_tokens=128,
)
print(runtime.answer("/path/to/image.jpg", "What is shown in this image?"))
```

The question is plain text: the loader adds the image token and Vicuna chat
format. The first load downloads the pinned Vicuna and CLIP weights unless they
are already cached. A local delta bundle alone is insufficient for offline use;
set `local_files_only=True` after all three model artifacts are cached. For a
Hub bundle, replace the local path with its repository ID and set `revision` to
the published commit hash. Keep tokens in your Hugging Face login or environment,
never in scripts or model cards.

Use `invllava predict --backend release` for the benchmark workflow. Keep its
runtime controls equal to those of the reference evaluation. Merging LoRA is
an optional inference optimization that needs its own numerical check.

Before uploading a model, document its applicable model license separately
from the code license, review its model card, check upstream access requirements,
and verify a fresh-cache download.
The HD and 13B configurations require separate model bundles. Label subset-trained
checkpoints with their training fraction and do not present them as full-data models.

## Verify a release

From a clean checkout, install the package and test both the local bundle and
the downloaded Hub snapshot. The fixed-input check uses existing prepared
benchmark examples and needs no new training:

```bash
python scripts/verify_release_inference.py \
  --release /workspace/exports/Inverse-LLaVA-7B \
  --examples /workspace/data/ocrbench/examples.jsonl \
  --cache-dir /workspace/cache \
  --samples 4 --output /workspace/reports/release-inference.json
```

This checks checksums, finite logits, repeatable greedy answers, image-token
placement and sensitivity to visual features. Its 16-token generations are
diagnostics, not benchmark scores. Use the [evaluation guide](evaluation.md) for
complete benchmark runs. Verify a new Hub download against the local bundle's
weight hash and outputs before adding its link to the README.

Publish the compact weights, configuration, checksums, model card and model
terms on Hugging Face. Keep source and usage examples on GitHub. Selected
prediction records, scores, protocol configurations and figure-source tensors
can be distributed separately with their sample IDs, hashes and data terms.
Optimizer state, caches, credentials, manuscript drafts and restricted dataset
images do not belong in the inference bundle.
