# Data preparation

Prepare the full datasets on a Linux training host with adequate persistent
storage. The paths below use `/workspace` as an example. Keep models and
datasets outside the source checkout.

## 1. Freeze and acquire

First inspect the plan; this command never downloads:

```bash
uv run --no-sync invllava data plan configs/data/llava_mix665k.yaml --destination /workspace/data
```

Replace every `pending-freeze` with an immutable provider revision before
acquisition. Record archive SHA-256 values in the source ledger. The guarded
fetch command handles registered Hugging Face files and reports manual sources
without pretending to fetch them:

```bash
INVLLAVA_ALLOW_DOWNLOADS=1 uv run --no-sync invllava data fetch \
  configs/data/llava_mix665k.yaml --destination /workspace/data --allow-download
```

For a first acquisition whose digest is not yet known, use the separate
candidate lane. It writes a sidecar with the observed URL, size, provider
validators, and SHA-256; it does not make the artifact scientifically trusted:

```bash
INVLLAVA_ALLOW_DOWNLOADS=1 uv run --no-sync invllava data acquire-candidate \
  <official-url> --destination /workspace/downloads/<archive> \
  --minimum-free-gib 100 --allow-download
```

Review the sidecar, independently check the official endpoint, copy its digest
and stable release identity into the data config, then reacquire or verify it
through `data fetch`. COCO's historical official endpoint is HTTP-only (its
HTTPS certificate does not match the image host); only that candidate command
uses the explicit `--allow-insecure-http` exception. Once its SHA-256 is frozen,
all later downloads remain content-authenticated by the digest.

Acquire COCO train2017, GQA images, TextVQA train/validation images, and both
Visual Genome image parts from their official providers. This source set
follows the [official LLaVA-1.5 training
instructions](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/README.md#train).
The exact endpoints are recorded in `configs/data/llava_mix665k.yaml`.

OCR-VQA needs an explicit provenance qualifier. Its original URL-based image
acquisition is no longer complete, so the config pins a checksummed community
snapshot already arranged for the LLaVA-1.5 mixture. We call this source the
**OCR-VQA reconstruction** in configs, manifests, tables, and run records. It
is suitable for controlled experiments in which every condition uses the same
bytes; it is not described as the canonical OCR-VQA release. The pinned
archive has been fully CRC-read, all 80,000 referenced images have been decoded,
and its complete-mixture inventory is recorded below. Do not commit data or
provider credentials.

## 2. Verify, extract, and arrange

Extract each downloaded archive through the verified, path-safe interface. The
destination must not already exist, so a failed extraction cannot look complete:

```bash
uv run --no-sync invllava data extract /workspace/downloads/train2017.zip \
  --destination /workspace/data/llava-v1.5-mix665k/components/coco \
  --sha256 <64-lowercase-hex> --minimum-free-gib 100
```

The COCO archive already contains its `train2017/` directory, so extracting it
into a path also named `train2017` would create the incorrect nested path
`train2017/train2017`. Inspect every verified archive's top-level entries before
choosing its component destination. The reviewed component contract is:

```text
components/
├── coco/train2017/
├── gqa/images/
├── ocr_vqa/images/
├── textvqa/train_images/
├── vg-part1/VG_100K/
└── vg-part2/VG_100K_2/
```

After every component archive has been digest/CRC verified and extracted,
assemble the annotation-facing root without duplicating image payloads:

```bash
uv run --no-sync invllava data assemble-layout \
  configs/data_layout/llava_mix665k.yaml \
  --component-root /workspace/data/llava-v1.5-mix665k/components \
  --destination /workspace/data/llava-v1.5-mix665k/raw \
  --manifest /workspace/data/llava-v1.5-mix665k/layout.manifest.json
```

The resulting image root matches paths stored by the official annotation:

```text
raw/
├── coco/train2017/
├── gqa/images/
├── ocr_vqa/images/
├── textvqa/train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

Extraction rejects path traversal, links/special files, unknown formats, an
existing destination, and archives whose declared expansion would violate the
free-space reserve. ZIP extraction also enforces each member's CRC. Layout
assembly rejects links, overlapping destinations, cross-filesystem hardlinks,
and component/output nesting; its manifest records file/byte inventories and
the layout-config digest. Preserve the acquisition sidecar, and delete a source
archive only after its digest, full archive/CRC read, expected extraction
topology, and extracted inventory pass. Keep the extracted component through
the source-wise and complete image audits, final layout, normalization, and
prepared-manifest validation. Use a distinct root for the 558K paired pool
rather than mixing it into the 665K tree.
The official 558K ZIP expands its numeric shard directories directly under the
chosen extraction destination; it does not add an `images/` wrapper directory.

## 3. Audit before normalization

First stream through the complete annotation. This catches malformed JSON,
unsafe image paths, invalid roles/turn order, placeholder mismatches, reports
reused source IDs, and rejects a wrong annotation digest or unexpected sample
count without loading the full JSON array into memory:

```bash
uv run --no-sync invllava data audit-annotation \
  /workspace/data/llava-v1.5-mix665k/sources/llava-v1.5-mix665k-annotation/llava_v1_5_mix665k.json \
  --image-root /workspace/data/llava-v1.5-mix665k/raw \
  --source-revision <annotation-commit> \
  --expected-sha256 <annotation-sha256> \
  --expected-samples <verified-count> \
  --output /workspace/data/llava-v1.5-mix665k/annotation-audit.json
```

After every image source has been extracted into the final layout, hash and
fully decode every unique referenced image:

```bash
uv run --no-sync invllava data audit-images \
  /workspace/data/llava-v1.5-mix665k/sources/llava-v1.5-mix665k-annotation/llava_v1_5_mix665k.json \
  --image-root /workspace/data/llava-v1.5-mix665k/raw \
  --source-revision <annotation-commit> \
  --workers 16 \
  --inventory-output /workspace/data/llava-v1.5-mix665k/image-inventory.jsonl \
  --output /workspace/data/llava-v1.5-mix665k/image-integrity.json
```

This is deliberately stronger than checking `Image.open`: it verifies the
container, reopens and loads all pixels, exercises RGB conversion, and hashes
the encoded bytes. It explicitly disables Pillow's permissive truncated-image
mode, even if another imported library enabled it. It reports missing,
zero-byte, truncated/corrupt, and
zero-dimension files; format counts; and a deterministic inventory digest. A
failed report is written before the command exits nonzero, with the first 100
failures and a digest over the complete failure set. For the multi-corpus 665K
mixture, run a `--source coco`, `--source gqa`, `--source ocr_vqa`,
`--source textvqa`, or `--source vg` audit as soon as the corresponding logical
source is available. Visual Genome needs both archive parts before its one `vg`
source audit. After all sources pass individually, run the unfiltered audit
shown above. A source-filtered report is useful for early failure localization
but cannot authorize normalization.

The verified official 665K annotation contains 665,298 rows: 624,610 image
references and 40,688 text-only rows. Its constituent corpora reuse source IDs
(275,576 occurrences beyond the first), which is valid upstream structure and
is reported rather than rejected. Normalization never drops or deduplicates
these rows. It preserves their order, images, prompts, and targets, and assigns
deterministic `row-<index>:<source-id>` internal IDs so sampling and checkpoint
state are unambiguous. Use `--require-unique-ids` only when a source contract
explicitly promises a globally unique ID namespace.

For a single-corpus annotation whose image paths begin with storage shard names
rather than semantic corpus names (the official 558K pool uses numeric
directories), pass `--default-source llava-pretrain-558k-paired` consistently
to `audit-annotation`, `audit-images`, and `normalize`. Leave it unset for the
665K mixture so its `coco`, `gqa`, `ocr_vqa`, `textvqa`, `vg`, and text-only
`llava` sources remain distinguishable.

On failure, do not drop examples or enable truncated-image loading. Preserve
the small report, reacquire the named official source, verify its archive digest,
and repeat that source audit. Any later change to an image invalidates the
inventory digest and requires a new complete audit and prepared manifest.

For paired-supervision studies, write the optional content inventory during the
existing decode pass and compare it with each evaluation inventory before
training. Paths and encodings may differ; the comparison uses canonical RGB
pixel SHA-256 values:

```bash
invllava data compare-image-inventories \
  --left /workspace/data/llava-pretrain-558k-paired/image-inventory.jsonl \
  --right /workspace/data/benchmarks/ocrbench/image-inventory.jsonl \
  --require-disjoint \
  --output /workspace/exports/558k-vs-ocrbench-overlap.json
```

The immutable report binds both inventory hashes, corpus sizes, unique content
counts, total shared content, and bounded example paths. Exact disjointness is a
gate for continuation experiments. A separate perceptual-hash
or embedding review can screen near duplicates without changing this exact
content test.

Test the image-audit gate with a known truncated image before trusting a new
environment. Audit the complete annotation and every referenced image before
normalization. The 665K reference package contains 665,298 rows, 624,610 image
references and 349,034 unique image files. Store exact digests and decode counts
in the prepared manifest; validate each newly assembled copy independently.

## 4. Normalize and bind the audit

```bash
uv run --no-sync invllava data normalize \
  /workspace/data/llava-v1.5-mix665k/sources/llava-v1.5-mix665k-annotation/llava_v1_5_mix665k.json \
  --image-root /workspace/data/llava-v1.5-mix665k/raw \
  --image-audit /workspace/data/llava-v1.5-mix665k/image-integrity.json \
  --output /workspace/data/llava-v1.5-mix665k/train.jsonl \
  --manifest /workspace/data/llava-v1.5-mix665k/train.manifest.json \
  --data-id llava-v1.5-mix665k \
  --source-revision <annotation-commit>
```

Image-bearing data cannot be normalized for a scientific run without a passing,
complete `--image-audit` for the same annotation digest, revision, image root,
and reference count. The prepared manifest embeds that evidence and its report
digest. Training rejects a manifest with missing, failed, mismatched, or
unbound source-filtered image evidence before loading model weights. The explicit
`--allow-unverified-images` escape hatch is only for diagnostic fixture
construction and its output is not accepted by training when it contains image
references. Text-only examples in the official mixture remain valid and bypass
the vision encoder.

Controlled source-mixture experiments use `include_sources` in the data config
and repeat the same values through `--source` for the image audit and
normalization commands. The resulting schema-5 manifest binds the full parent
annotation digest, exact source filter, filtered sample audit, and matching
filtered image inventory. Training rejects any disagreement between them. This
allows a study to stage only its required image corpora while preserving the
original row indices used to qualify repeated source IDs.

Schema-4 normalized files store each image path relative to the JSONL parent,
which is the prepared package root. Move the JSONL, manifest, and image tree
together; the loader resolves paths from the JSONL location and rejects paths
that escape it. Earlier schema-3 manifests with absolute paths remain readable,
but should be regenerated before transfer to a new host. Check row equivalence
and sample-image bindings after relocation.

The 5% and 20% screens are deterministic, source-stratified, nested subsets of
this one normalized file; each run records its exact selected IDs. The 665K
instruction set and 558K paired pool are different treatments. The equal-row
instruction continuation matches optimizer updates. A second instruction
control matches supervised-token exposure to the nearest whole sample.

Materialize the additional causal controls once, after both parent packages and
the Vicuna tokenizer are present on the same filesystem:

```bash
python scripts/materialize_continuation_controls.py \
  --paired-jsonl /workspace/data/llava-pretrain-558k-paired/train.jsonl \
  --paired-manifest /workspace/data/llava-pretrain-558k-paired/train.manifest.json \
  --instruction-jsonl /workspace/data/llava-v1.5-mix665k/train.jsonl \
  --instruction-manifest /workspace/data/llava-v1.5-mix665k/train.manifest.json \
  --output-root /workspace/data/continuation-controls \
  --rows 5580 --seed 17 --workers 16
```

This creates independently sealed token-matched instruction, equal-update 50/50
paired/instruction, and shuffled-image correspondence packages. Images are
hard-linked, so source and output must share a filesystem. Each package records
both parent manifest hashes, selected IDs, supervised-token count, complete
selected-image decode audit, and materializer revision. Natural responses
cannot generally match update count and token exposure in one condition; the
two instruction controls answer those questions separately.

For a two-example end-to-end contract only, run
`scripts/create_synthetic_fixture.py`; it creates a normalized JSONL, prepared
manifest, passing image-integrity report, images, and evaluation fixture. It is
never scientific evidence.
