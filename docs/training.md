# Training

The reference recipe is `configs/experiment/canonical_7b.yaml`. It jointly
trains LoRA and fusion parameters for one epoch on LLaVA's 665K instruction
mixture. Vicuna and the CLIP encoder are frozen. Global batch size is 128,
learning rate 2e-4, LoRA rank 128, alpha 256 and dropout 0.05.

## Prepare data

Follow [Data preparation](guides/DATA_PREPARATION.md) in order: download pinned
annotations and image archives, verify downloads, extract each component,
assemble the expected layout, decode-check all referenced images, and normalize
the conversations. `data plan` only lists requirements; it does not download or
prepare the dataset. Explicit download opt-in is required.

Keep the prepared JSONL, manifest and image tree together. Failed images must
be investigated before training; silently dropping examples changes the recipe.
Use separate output directories for instruction data and the 558K paired pool.

## Launch

```bash
python scripts/launch_train.py \
  configs/experiment/canonical_7b.yaml \
  /workspace/data/llava-v1.5-mix665k/train.jsonl \
  /workspace/data/llava-v1.5-mix665k/train.manifest.json \
  --runtime-ref cuda_2gpu_zero2 --microbatch-size 32 \
  --gradient-checkpointing on \
  --run-root /workspace/runs --run-id inverse-llava-7b-seed42
```

This uses two GPUs and two accumulation steps. Keep microbatch grouping fixed
for strict comparisons: token-averaged losses can weight examples differently
when variable-length targets are regrouped. Changing hardware does not require
changing the architecture or dataset.

Before a full run, test a bounded subset with `--maximum-samples`, verify that
losses and gradients are finite, and compare uninterrupted and resumed training.
Store smoke tests separately from reported experiments.

## Resume and retain

Use `--resume-from` with a complete checkpoint from the same run. This restores
optimizer, scheduler, random-number and sampling state. `--initial-checkpoint`
starts a new continuation experiment and does not resume its parent's schedule.

```bash
invllava verify-run /workspace/runs/inverse-llava-7b-seed42 \
  --output /workspace/reports/inverse-llava-7b-integrity.json
```

Keep the resolved recipe, manifests, raw metrics, complete checkpoints and
verification report. Local JSONL metrics are the durable record; TensorBoard
and optional W&B provide views. Neither replaces checkpoint storage.

## Variants

The [ablation definitions](method/ABLATION_DEFINITIONS.md) specify what changes
and what stays fixed in each comparison. HD changes visual feature width;
13B changes the language backbone. Paired-data continuations are separate
experiments, with their source checkpoint and selected sample IDs recorded.
Select comparable checkpoints by exposure or a declared validation rule.
