# Evaluation

The [evaluation guide](guides/EVALUATION.md) contains data materialization,
prediction and scoring commands. The [protocol registry](benchmarks/INDEX.md)
identifies the frozen datasets, evaluators and validation checks.

## Use one protocol for every model

Prepare one benchmark package, validate its images and prompts, then reuse it
for Inverse-LLaVA and the official LLaVA-1.5 LoRA/FFT references. Each prediction
records the checkpoint, exact prompt, decoding settings and sample identity.
Score the final prediction file with the benchmark configuration used to create
it; retain both the raw predictions and score manifest.

`configs/reproduction/full.yaml` lists the primary and supplementary protocols.
Its MMBench configurations are `mmbench_en_llava.yaml` and
`mmbench_cn_llava.yaml`; MM-Vet uses `mmvet_gpt41_hosted.yaml`. The alternative
VLMEvalKit prompts and GPT-4-0613 judge remain separate configurations.

Run `invllava reproduction audit --stage code --strict` to check configuration
references without downloading data or loading a model. Environment and
scientific checks also require your execution-image record and runtime evidence;
a source checkout alone cannot establish completed benchmark results.

The primary evaluations use BF16, SDPA, KV caching, deterministic generation and
unmerged LoRA. A changed image processor, CLIP layer, prompt, precision or LoRA
execution mode is a separate protocol variant. Revalidate numerical behavior
before comparing its results.

## Local and external scoring

| Endpoint | Scoring |
|---|---|
| GQA, ScienceQA-IMG, TextVQA, MME | Local, public references |
| MMBench EN/CN public development splits | Local circular evaluation |
| VizWiz released test package | Local public annotations under the documented join |
| VQAv2 validation | Local official consensus metric |
| VQAv2 test-dev | EvalAI submission; hidden labels |
| MM-Vet | Official hosted judge; record model/version and responses |
| OCRBench, AI2D, MMStar | Local supplementary benchmarks |

Public annotations permit local scoring only for the corresponding split and
protocol. VQAv2 validation cannot stand in for a test-dev result. Development
subsets establish interface correctness; report full-benchmark results separately.

## Comparisons and uncertainty

Disclosed paper scores and locally evaluated checkpoints should occupy separate
rows. Official FFT and LoRA references differ in training; use training-matched
controls for architectural attribution. Report the judge identity for MM-Vet.

Use paired resampling of the same evaluation units to quantify differences.
For MME, preserve its image-level pairs and category structure. Evaluation
confidence intervals do not measure variation across independent training seeds.
