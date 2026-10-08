# Representation and qualitative studies

Use matched inputs and declared feature locations for every comparison. Keep
high-dimensional quantitative measurements alongside dimensionality reductions;
2D plots alone do not establish alignment quality or explain benchmark scores.
See [Analysis](../analysis.md) for interpretation and presentation conventions.

## Immutable capture workflow

Use a frozen normalized example file with exactly one image per row. The
capture command uses the same release or reference loader as evaluation and
writes sample IDs inside the NPZ plus a JSON sidecar containing hashes, shapes,
input condition, dtype, and pooling policy:

```bash
INVLLAVA_ALLOW_DOWNLOADS=1 invllava capture-representations \
  --backend release --model /models/Inverse-LLaVA-7B \
  --cache-dir /workspace/cache \
  --examples /data/eval/representation_panel.jsonl \
  --batch-size 4 --output /runs/analysis/inverse-representations.npz \
  --allow-download
```

Arrays include selected native vision features, every pooled LLM hidden state,
the terminal prompt state at each layer, and—for Inverse checkpoints—the Q/K/V
mapped-text and pre-output joint representations at each fusion layer. Native
controlled LLaVA and HF LLaVA captures additionally include early projected
visual features. The common cross-model key is `hidden.last.<layer>`; it avoids
equating Inverse's separate visual branch with LLaVA's inserted patch tokens.
Multi-image examples fail instead of being silently averaged.

Compare any two arrays only when their ordered sample IDs match:

```bash
invllava compare-representations \
  --left /runs/analysis/inverse-representations.npz \
  --left-key fusion.0.q.mapped_text \
  --right /runs/analysis/inverse-representations.npz \
  --right-key vision.selected --k 10 \
  --matched-margin --matched-permutations 10000 \
  --confidence 0.95 --seed 2026 \
  --output /runs/analysis/inverse-mapped-q-vs-vision.json
```

This reports linear CKA, RSA Spearman, kNN overlap, and both effective-rank
definitions, plus normalized singular-value spectra. Add `--matched-margin`
only when the two arrays have equal feature width and represent paired text and
visual features. `--matched-permutations` compares the observed mean paired
cosine with a deterministic shuffled-assignment null distribution and reports
its interval and one-sided permutation p-value. Treat the single-shuffle margin
as a diagnostic; use the permutation result for a paper-facing correspondence
claim.

Generate the planned layerwise CKA/effective-rank panel directly from two
aligned artifacts:

```bash
invllava plot-representation-cka \
  --left /runs/analysis/inverse-representations.npz \
  --right /runs/analysis/llava-representations.npz \
  --left-prefix hidden.last. --right-prefix hidden.last. \
  --output /runs/analysis/layerwise-cka.pdf
```

For same-width spaces, fit one joint PCA basis rather than unrelated bases:

```bash
invllava plot-representation-pca \
  --series native /runs/analysis/inverse-representations.npz vision.selected \
  --series mapped-q /runs/analysis/inverse-representations.npz fusion.0.q.mapped_text \
  --output /runs/analysis/native-vs-mapped-pca.pdf
```

Both commands require identical ordered sample IDs, refuse overwrites, and save
a JSON sidecar with source and figure hashes. Freeze key prefixes before looking
at task correlations.

After local scoring has produced `details.per_item`, select qualitative IDs by
the fixed correctness partition:

```bash
invllava select-cases \
  --model inverse /runs/eval/inverse.jsonl /runs/eval/inverse-score.json \
  --model llava-lora /runs/eval/llava-lora.jsonl /runs/eval/llava-lora-score.json \
  --model llava-fft /runs/eval/llava-fft.jsonl /runs/eval/llava-fft-score.json \
  --primary inverse \
  --per-group 4 --seed 2026 --output /runs/analysis/case-ids.json
```

The output freezes the selected IDs. Error labels and captions remain an
auditable human annotation step; shown outputs stay verbatim. It also records
the full candidate count and displayed count for every correctness pattern, so
the case grid cannot be read as an estimate of pattern frequency.

Export the selected IDs into a hashed JSON record and an optional Markdown
review sheet containing exact prompts, references, image paths, and unedited
outputs:

```bash
invllava export-cases \
  --selection /runs/analysis/case-ids.json \
  --examples /data/eval/case-panel.jsonl \
  --prediction inverse /runs/eval/inverse.jsonl \
  --prediction llava-lora /runs/eval/llava-lora.jsonl \
  --prediction llava-fft /runs/eval/llava-fft.jsonl \
  --output /runs/analysis/cases.json \
  --markdown /runs/analysis/cases.md \
  --image-root /runs/analysis/case-images \
  --examples-output /runs/analysis/case-examples.jsonl
```

`--image-root` copies only selected images into a content-addressed directory
and records their hashes in the case JSON. This keeps case evidence usable after
the full benchmark dataset and disposable cache are removed. The optional
`--examples-output` is a self-contained evaluation file for the same IDs and
images, suitable for the intervention commands below.

For image-dependence evidence, materialize deterministic deranged-image and
fixed-RGB blank-image versions of the same panel. They preserve IDs, prompts,
references, and scoring protocol, and record intervention provenance in both
sample metadata and a sidecar manifest:

Shuffling operates on distinct decoded-RGB images. All questions sharing an
image receive the same replacement, and every replacement has different pixels.
This preserves MME question pairs and handles duplicate images saved under
different paths. At least two distinct images are required. The manifest records
the image-group count and policy; sample metadata binds original and replacement
RGB hashes. A fixed seed reproduces the mapping for the same ordered input.
Different pixels can still contain answer-relevant content; a shuffled image
does not guarantee that the original answer becomes false.

```bash
invllava prepare-interventions --examples /data/eval/case-panel.jsonl \
  --mode shuffled --seed 2026 \
  --output /runs/analysis/case-panel-shuffled.jsonl

invllava prepare-interventions --examples /data/eval/case-panel.jsonl \
  --mode blank --blank-rgb 127 127 127 \
  --image-root /runs/analysis/blank-images \
  --output /runs/analysis/case-panel-blank.jsonl
```

Run ordinary immutable prediction and scoring on each file, then summarize the
paired correctness and response changes with:

```bash
invllava intervention-summary \
  --original-score /runs/analysis/original-score.json \
  --intervened-score /runs/analysis/blank-score.json \
  --original-predictions /runs/analysis/original.jsonl \
  --intervened-predictions /runs/analysis/blank.jsonl \
  --output /runs/analysis/blank-summary.json
```

The response comparison records exact changes and a whitespace/case-normalized
view while preserving all generated text in the source prediction files. Do not interpret
shuffled or blank performance as a localization map; patch occlusion remains a
separate analysis specified before capture if it is later added.
