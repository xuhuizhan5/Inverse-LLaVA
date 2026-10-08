# Benchmark protocol registry

[Benchmark configurations](../../configs/benchmark/) pin each split, source,
prompt, decoding policy, and scorer. The
[evaluation guide](../guides/EVALUATION.md) provides preparation, prediction,
scoring, and submission commands.

| Benchmark | Configuration | Scoring and protocol |
|---|---|---|
| ScienceQA-IMG | `scienceqa_img.yaml` | Local; LLaVA CQM-A prompts and image-only questions |
| MMBench EN/CN | `mmbench_en_llava.yaml`, `mmbench_cn_llava.yaml` | Local public v1.0 dev; LLaVA prompts and circular scoring, without judge fallback |
| MME perception/cognition | `mme_perception.yaml`, `mme_cognition.yaml` | Local; preserve image-level question pairs and official category aggregation |
| MM-Vet | `mmvet_gpt41_hosted.yaml` | Local generation and packaging; pinned official hosted GPT-4.1 grading |
| VizWiz | `vizwiz.yaml` | Local; released LLaVA prompts joined to public test annotations |
| VQAv2 validation | `vqav2_val.yaml` | Local official consensus score; a development split, separate from test-dev |
| VQAv2 test-dev | `vqav2_testdev.yaml` | External hidden-label scoring; retain the accepted submission and server receipt |
| TextVQA validation | `textvqa.yaml` | Local; Rosetta OCR-conditioned prompts and EvalAI normalization |
| GQA balanced test-dev | `gqa.yaml` | Local; pinned LLaVA question fixture and official scorer |
| AI2D | `ai2d.yaml` | Local supplementary evaluation |
| MMStar | `mmstar.yaml` | Local supplementary evaluation with macro-average over L2 capabilities |
| OCRBench | `ocrbench.yaml` | Local supplementary evaluation with category-specific answer handling |
| MathVista testmini | `mathvista_testmini.yaml` | Configuration only; preparation and extraction remain unavailable pending reference verification |

The primary comparison uses the configurations above. `mmbench_en.yaml` and
`mmbench_cn.yaml` provide alternate VLMEvalKit prompts; `mmvet.yaml` defines
the separate GPT-4-0613 judge. Keep outputs from these alternatives distinct
from the primary protocols.

## Verification

Each configuration records its verification status. Reference-scorer checks
validate the adapter; every new model evaluation still needs complete sample
coverage, matching prediction manifests, and a saved score or service receipt.

Before reusing ScienceQA inputs, run `scripts/verify_scienceqa_golden.py`
against the checksummed `llava_test_CQM-A.json` member of the official LLaVA
evaluation archive. It checks every image-question ID, prompt, and reference.
The [official launcher](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/sqa.sh)
uses this artifact with `--single-pred-prompt`.

For MME, compare new answer files with the original calculator. The pinned
lmms-eval extractor additionally accepts y/n and strips punctuation, so parser
parity must be checked on the actual predictions. For hosted evaluation,
record the judge or service version and archive the raw returned grades.
