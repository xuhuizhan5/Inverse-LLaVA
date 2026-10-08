# Upstream sources and version locks

Source identities live with the configuration that consumes them. Use these
records when downloading data, loading models, or checking an evaluator.

| Component | Authoritative record |
|---|---|
| Python dependencies | [uv.lock](../uv.lock) and [requirements.txt](../requirements.txt) |
| CUDA/PyTorch base image | [Immutable image reference](../containers/base-image.digest) |
| Language and vision models | [Model configurations](../configs/model/) |
| Training annotations and image archives | [Dataset configurations](../configs/data/) |
| Benchmark questions, prompts, images, and scorers | [Benchmark configurations](../configs/benchmark/) |
| Official-scorer comparisons | [Golden-test guidance](../tests/golden/README.md) and protocol tests under [tests/unit](../tests/unit/) |

The reference 7B configuration pins `lmsys/vicuna-7b-v1.5` at
`3321f76e3f527bd14065daf69dad9344000a201d` and
`openai/clip-vit-large-patch14-336` at
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`.
LLaVA prompt and converter references use commit
`c121f0432da27facab705978f83c4ada465e46fd`.
The benchmark cards include per-file SHA-256 values and, where applicable,
the LMMS-Eval or VLMEvalKit reference revisions.

## Data provenance

[The 665K instruction configuration](../configs/data/llava_mix665k.yaml)
pins the annotation revision and each image archive independently. The
annotation contains 665,298 rows. OCR-VQA images use the explicitly identified
LLaVA-layout reconstruction; this is not a claim of byte identity with the
original URL-based OCR-VQA release. Preserve that source distinction in
training manifests and comparisons.

[The 558K paired-data configuration](../configs/data/llava_pretrain558k.yaml)
pins the separate image-text dataset. Additional paired-data training is a
distinct experiment from the one-stage instruction-only reference.

## Verification and terms

A source hash establishes file identity, not permission to redistribute.
Model weights and datasets retain their upstream terms; see
[third-party notices](../THIRD_PARTY_NOTICES.md).
For a new download, verify the configured hash, decode all referenced images,
and retain the resulting audit. For evaluation, retain the protocol revision,
prediction coverage checks, raw predictions, and the scorer or server receipt.
The [evaluation guide](../docs/evaluation.md) describes local and external
scoring paths.

The answer-normalization reference in LLaVA traces to Pythia/MMF revision
`c46b3b3391275b4181567db80943473a89ab98ab`. Its
[BSD license](licenses/Pythia-BSD.txt) is retained with the source.
