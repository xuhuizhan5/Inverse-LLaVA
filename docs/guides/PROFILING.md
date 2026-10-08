# Profiling protocol

The portable SDPA runtime is the publication reference. The optional
TorchInductor/Triton lane and its mandatory numerical gates are documented in
[Optional kernel optimization](KERNEL_OPTIMIZATION.md). Do not compare a cold
compiled call with a warmed portable call or report a tiny-model audit as 7B
throughput.

Compare controlled LLaVA-LoRA and Inverse on the same GPU model, container, precision,
batch, prompt/image lengths, and decoding length. Warm up at least 10 iterations;
measure at least 30 synchronized repetitions and report median/IQR.

Record total/trainable parameters, calculated MACs, peak allocated/reserved
memory, first-token latency, decode tokens/s, examples/s at batch 1 and 4, and
100–300 steady training steps. Label every datum `measured`, `calculated`,
`extrapolated`, or `reported`. Do not infer the avoided alignment-stage wall time
from an inference profile or rerun the full LLaVA alignment stage solely to time
it.

Before a larger model's subset run, use `scripts/verify_training_memory.py`
with its frozen experiment and initialization. Launch through
`torch.distributed.run` with the declared process count and allocator environment.
The probe uses the existing training engine, optimizer, accumulation, precision,
and checkpoint path for two full-length synthetic updates. Run image and text
workloads separately. The first update is unpadded; subsequent updates pad one
trailing token in alternating rows, exercising the explicit attention-mask path.
It verifies actual expanded-token exposure and finite
optimization, including allocated optimizer state. Its output is operational
evidence: synthetic loss, throughput, and weights must not enter result tables
or be released as a trained model. Keep the external timeout/shutdown guard.

For the reviewed two-GPU runtime and a saved, hash-bound initialization:

```bash
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.75,expandable_segments:True \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  scripts/verify_training_memory.py \
  --experiment /runs/initialization/experiment.yaml \
  --initial-checkpoint /runs/initialization \
  --modality image --updates 2 --output-dir /runs/diagnostics/memory-image
```

Repeat with `--modality text` and a new output directory. Stage weights first;
the command is offline and refuses to overwrite an existing diagnostic.

Run the same frozen example panel and command for the controlled LLaVA-LoRA and
Inverse checkpoints:

```bash
invllava profile-native \
  configs/experiment/canonical_7b.yaml \
  --checkpoint /runs/canonical/checkpoints/step-0005198 \
  --examples /data/eval/profile-panel.jsonl \
  --batch-sizes 1 4 --decode-tokens 32 \
  --warmups 10 --repetitions 30 --include-preparation \
  --output /runs/profiles/inverse.json
```

The autoregressive measurement prepares image/prompt inputs once, then reports
model-only time to first token and a fixed cached decode with EOS ignored so both
architectures perform equal work. `--include-preparation` separately measures
image loading, preprocessing, vision encoding, projection/fusion preparation,
and sequence expansion. The JSON also records hashes, hardware, software,
precision, attention backend, and total/trainable parameters. Do not compare two
files unless every field except architecture/checkpoint identity matches.

Create the comparison from sealed profiles. The plotting command
refuses mixed hardware, precision, attention backend, decode policy, or batch
sets and writes a source-hash sidecar:

```bash
invllava plot-profile-comparison \
  --profile Inverse /runs/profiles/inverse.json \
  --profile LLaVA-LoRA /runs/profiles/llava-lora.json \
  --output /runs/profiles/inference-comparison.pdf
```

The figure reports decode throughput, time to first token, and allocated memory
at each measured batch size. Keep parameter counts and calculated MACs in the
associated table because they have different units and scaling assumptions.

For the official Hugging Face FFT reference, use the isolated adapter on the
fully local indexed checkpoint:

```bash
invllava profile-hf-llava \
  --checkpoint /checkpoints/llava-v1.5-7b-hf-fft \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --examples /data/eval/profile-panel.jsonl \
  --batch-sizes 1 4 --decode-tokens 32 \
  --warmups 10 --repetitions 30 --include-preparation \
  --output /runs/profiles/llava-fft.json
```

Before timing, the command compares direct HF logits, the first token, and one
cache-continuation step with the adapter's expanded `inputs_embeds` route. It
records the numerical tolerances and refuses to emit a profile when parity
fails. The HF artifact labels its runtime `requires_grad` count separately from
a training-time trainable-parameter count because checkpoint loading does not
recover the original optimizer freeze policy. Run all timing files on an
otherwise idle GPU.
