# lmms-eval interoperability

Inference uses this repository's native model loaders and evaluation commands.
Pinned lmms-eval task utilities serve as independent reference scorers for
several benchmark checks. An end-to-end Inverse-LLaVA model adapter for
lmms-eval is not included.

The optional environment definition in `requirements/lmms-eval.in` is separate
from the training environment. Resolve its dependencies with hashes before
using `scripts/bootstrap_lmms_golden.sh`. The text-only
`src/invllava/eval/interop/lm_eval.py` adapter targets lm-evaluation-harness,
which is a different project.

## Benchmark compatibility

A matching task name does not guarantee a matching evaluation protocol.
Preserve the following contracts when adding another harness.

| Benchmark | Required contract |
|---|---|
| MME | Image-level question pairs, category aggregation, and parser parity with the official calculator |
| TextVQA | The exact Rosetta-OCR-conditioned LLaVA prompt and ten-reference consensus scoring |
| MMBench | Language, release, split, circular rotations, answer extraction, and judge policy |
| ScienceQA-IMG | CQM-A prompts, image-only filtering, option order, and official extraction |
| VizWiz | Released LLaVA question text joined to the public test answers, including unanswerable instructions |
| VQAv2 / MM-Vet | The specified hidden-label scoring service or hosted judge and its receipts |

The [benchmark registry](../benchmarks/INDEX.md) and
[benchmark configurations](../../configs/benchmark/) record the protocols.
Model, dataset, task code, and scorer versions must be pinned independently.

## Adding a model adapter

Use the existing `load_pretrained` interface and generation implementation.
The adapter should pass questions, images, and decoding settings without access
to answer references. Preserve Vicuna-v1 prompt rendering, token and image
placement, padding, CLIP features, stopping rules, and sample order. Official
LLaVA-LoRA checkpoints require their base model as well as adapter weights.

Compare a fixed panel through both execution paths before running a full
benchmark: sample IDs, prompt token IDs, processed pixels, generated answers,
per-item scores, and aggregate scores must agree under the declared numerical
policy. Test the official LLaVA reader independently when validating that
baseline. Retain raw harness logs and an explicit sample-ID mapping alongside
the native prediction manifests.

Select complete image-level pairs for MME subsets. Use image-clustered
resampling for VQA confidence intervals. These grouping rules must survive
conversion between harness formats.

See the upstream [model integration guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/guides/model_guide.md).
