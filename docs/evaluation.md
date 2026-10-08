# Evaluation

## Rescore the released answers

The [research artifact collection](https://huggingface.co/xuhuizhan5/Inverse-LLaVA-research-artifacts)
contains generated answers and score records for the published comparisons.
Its `evaluations/index.json` identifies each checkpoint, metric and protocol.
Download only the benchmark folders you need and prepare the official examples
as described below. Then, for example:

```bash
python scripts/rescore_published_answers.py configs/benchmark/textvqa.yaml \
  --answers /path/to/evaluations/textvqa/inverse-llava/answers.jsonl \
  --examples /workspace/data/textvqa/examples.jsonl \
  --output /workspace/reports/textvqa-rescored.json
```

This verifies complete sample coverage and every prompt hash before applying
the pinned scorer. The export retains original inference identifiers separately
from the current scoring identifiers. All 33 local comparison cells were
rescored from these exports and matched the recorded scores. VQAv2 test-dev
and hosted MM-Vet retain their external results; this script does not replace
their official scoring procedures. Dataset images, questions and annotations
must come from the official sources.

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
