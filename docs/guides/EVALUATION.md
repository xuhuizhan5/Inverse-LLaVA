# Evaluation workflow

Freeze split, sample IDs, prompt, decoding, extraction, scorer revision, and
checkpoint before inference. Generate immutable JSONL once, then score repeatedly
without re-running the model. Golden-test adapters against official scorer output
before changing `verification_status`.

When run from a source checkout, `predict` records the imported checkout's
execution-source hash automatically and rejects a conflicting external stamp
before model loading. Git initialization is optional. The hash includes the
implementation, configurations, and dependency lock; credentials and datasets
are excluded. Wheel-based deployments must retain their build provenance and
supply `INVLLAVA_EXECUTION_SOURCE_SHA256` explicitly when no clean source commit
is available. External packaging rejects results with neither source identity.

For the nine primary endpoints, start with the [evaluation summary](../evaluation.md). The
[lmms-eval guide](LMMS_EVAL.md) describes the reference-scorer checks and
requirements for an additional harness adapter. Model inference uses this
repository's native evaluation workflow.

For a completed MMBench run, inspect extraction and rotation errors without
changing its score:

```bash
python scripts/audit_mmbench_responses.py \
  --examples DATA/examples.jsonl --predictions RUN/predictions.jsonl \
  --score RUN/score.json --output REPORT/mmbench-responses.json
```

The audit verifies input hashes, IDs, prompts, and exact rescore agreement. It
reports base/all-rotation accuracy, circular accuracy, category results, and
unparsed IDs. Its parsing-only ceiling is a diagnostic bound, not a new score.

Local by default once goldens pass: ScienceQA-IMG, MMBench EN/CN public dev
answers, MME, VizWiz April 2026 released test answers, TextVQA val, GQA balanced
test-dev, AI2D, MMStar, and OCRBench. Optional MathVista testmini currently has
a protocol card but its converter/extraction path is intentionally blocked
pending an official golden. MM-Vet generation is local but its score is
judge-dependent. VQAv2 test-dev is the only paper protocol here that is
external-only because labels remain hidden; use VQAv2 val for development and
do not compare its value directly with test-dev.

For VizWiz, use the answer-bearing
`VizWiz_all_answers/VQA_test.json` linked by the official dataset page—not the
legacy `vqa_data/Annotations/test.json`, which contains questions but no
answers. Once the image-archive digest in the card is frozen, materialize it
with the same checksummed interface as the other registered benchmarks:

```bash
export INVLLAVA_ALLOW_DOWNLOADS=1
invllava fetch-eval configs/benchmark/vizwiz.yaml \
  --destination /workspace/data/eval/vizwiz \
  --cache-dir /workspace/cache/huggingface --allow-download
```

The materializer resumes the two HTTP sources, verifies both digests, extracts
only the 8,000 referenced images, fully decodes them, and records their encoded
and canonical-pixel hashes in a relocatable manifest.

Never silently replace official category rules with generic exact match. The
MMBench adapter applies deterministic VLMEvalKit-style choice extraction and
counts a base question correct only when every circular rotation is correct.
It intentionally has no LLM-judge fallback. OCRBench's category rules are
implemented without process-global state and match pinned lmms-eval outputs on
all 1,000 materialized examples plus deterministic perturbations. MMStar's
official MCQ extractor and macro-average over L2 capabilities match the pinned
reference over all 1,500 examples. MathVista remains blocked on its official
extraction golden.

For the original LLaVA prompt, use `mmbench_en_llava.yaml` and
`mmbench_cn_llava.yaml`. Their full dev packages have been compared with the
actual prompt-building statements in pinned LLaVA code, including the Chinese
letter-answer instruction and 1,024-token ceiling. The existing
`mmbench_en.yaml`/`mmbench_cn.yaml` retain the separately verified VLMEvalKit
prompt. These protocols must not share cached predictions. Each has 4,329
input rows and 1,164 circular scoring groups. Record source hashes and
complete prediction coverage for each evaluated checkpoint.

### MM-Vet

The official v1 archive contains 218 questions and 200 images, occupying about
67 MB after extraction. `prepare-eval` now verifies its annotation digest,
preserves the `v1_*` IDs and `<AND>`/`<OR>` references, and audits every image:

```bash
invllava prepare-eval configs/benchmark/mmvet_gpt41_hosted.yaml \
  --annotations /workspace/data/mm-vet/mm-vet.json \
  --image-root /workspace/data/mm-vet/images \
  --output /workspace/evaluations/mmvet-inputs/examples.jsonl
```

Manual-file preparation records absolute image paths; retain those mount paths
when moving these inputs. The full LLaVA prompt/ID/reference check has passed.
The official hosted submission/download path passes on the retained public
LLaVA 7B answer fixture. Retain each model's raw grade and service receipt.

Use the [official hosted evaluator](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator)
under the pinned `mmvet_gpt41_hosted.yaml` protocol. Its free option
currently uses GPT-4.1 with one grading run. `mmvet.yaml` retains the separate
GPT-4-0613 protocol. Label the judge explicitly in tables, grade every compared
model identically, retain the service revision and downloaded raw grade ZIP,
and check every returned ID/model/score. Public references do not make MM-Vet
an exact-match benchmark. An API failure must never become a scored model error.

Package completed predictions through the shared interface:

```bash
invllava package-submission configs/benchmark/mmvet_gpt41_hosted.yaml \
  --examples /workspace/evaluations/mmvet-inputs/examples.jsonl \
  --predictions /workspace/evaluations/mmvet/predictions.jsonl \
  --output /workspace/evaluations/mmvet/submission.json \
  --output-manifest /workspace/evaluations/mmvet/submission.manifest.json
```

During adapter qualification, explicitly add `--allow-unverified`; keep those
artifacts identified as qualification runs. The command requires complete IDs,
unchanged prompts, and prediction/input hashes matching the evaluation manifest.
For an HF reference, also pass `--checkpoint-snapshot` at its recorded evaluation
path. This verifies every snapshot file and records its inventory digest. Native
and Inverse release manifests already contain their model-delta digest.

Submit from a small isolated CPU environment, without installing Gradio into the
training environment:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-project --with gradio-client==2.6.1 \
  python scripts/mmvet_hosted.py \
  --submission /workspace/evaluations/mmvet/submission.json \
  --annotations /workspace/data/mm-vet/mm-vet.json \
  --output-dir /workspace/evaluations/mmvet/hosted-gpt41
```

This uses the official service's free GPT-4.1 option and sends no API key. The
client checks the pinned service revision and preserves the request, raw ZIP,
and all 218 item grades. `accepted-grade.json` stores the unrounded mean on
the 0–1 scale and actual judge model. A failed request produces `failure.json`;
retry into a new directory after diagnosis. The service does not expose API
response IDs or complete retry histories, so its receipt cannot rule out every
server-side failure. Visible missing, malformed, or inconsistent grades are
rejected. One judging run provides no estimate of judge-run variance.

Import a completed receipt through the standard scorer:

```bash
invllava score configs/benchmark/mmvet_gpt41_hosted.yaml \
  --examples /workspace/evaluations/mmvet-inputs/examples.jsonl \
  --predictions /workspace/evaluations/mmvet/predictions.jsonl \
  --submission /workspace/evaluations/mmvet/submission.json \
  --judge-dir /workspace/evaluations/mmvet/hosted-gpt41 \
  --output /workspace/evaluations/mmvet/score.json
```

During qualification this also requires `--allow-unverified`. Import checks
the exact answer bytes, complete IDs, service/reference revisions, dated judge
identity, archive digest, raw grades, and aggregate. Keep the raw ZIP when
moving results between hosts; its recorded absolute download path may change.

### Other original benchmarks

MME materialization streams the pinned 859.6 MB release archive, validates each
question against LLaVA's checksummed paper-era prompt fixture, and preserves the
two-question image groups required by accuracy-plus scoring. Perception and
cognition are separate local artifacts:

```bash
export INVLLAVA_ALLOW_DOWNLOADS=1
invllava fetch-eval configs/benchmark/mme_perception.yaml \
  --destination /workspace/data/eval/mme-perception \
  --cache-dir /workspace/cache/huggingface --allow-download
invllava fetch-eval configs/benchmark/mme_cognition.yaml \
  --destination /workspace/data/eval/mme-cognition \
  --cache-dir /workspace/cache/huggingface --allow-download
```

VQAv2 and TextVQA use distinct normalizers. VQAv2 mirrors the official
evaluator's conditional normalization; TextVQA applies the EvalAI answer
processor to every prediction and reference before consensus. TextVQA's full
5,000-example golden matches the released prompts, references, image bindings,
and pinned scorer, including exact reproduction of the released 61.254% score.
VQAv2 val matches the pinned official evaluator on five complete stress panels,
including 1,071,770 item scores and every answer/question-type aggregate. Its
checksummed materializer decodes all 40,504 COCO val2014 images:

```bash
export INVLLAVA_ALLOW_DOWNLOADS=1
invllava fetch-eval configs/benchmark/vqav2_val.yaml \
  --destination /workspace/data/eval/vqav2-val \
  --cache-dir /workspace/cache/huggingface --allow-download
```

Test-dev packaging follows LLaVA's separate upload procedure: its M4C answer
processor runs before server submission. Raw predictions remain unchanged in
the prediction artifact. The hidden-reference normalization described above
applies to direct official VQA scoring, including the local validation split.

After freezing the dataset revisions, AI2D, ScienceQA-IMG, MMStar, and OCRBench
can be materialized directly into the shared immutable local schema:

```bash
export INVLLAVA_ALLOW_DOWNLOADS=1
invllava fetch-eval configs/benchmark/ai2d.yaml \
  --destination /workspace/data/eval/ai2d \
  --cache-dir /workspace/cache/huggingface --allow-download
```

The command writes decoded images, `examples.jsonl`, `image-integrity.json`, and
a manifest with the protocol-config hash, exact dataset revision/fingerprint,
example hash, per-image hashes, and complete image-inventory digest. Its
Hugging Face converter covers these four task schemas plus TextVQA; MME uses its
checksummed streaming ZIP converter. Use a disposable cache during validation
and remove it through the bounded cleanup after the fixture passes.
Self-contained artifacts record images as paths relative to `examples.jsonl`;
the loader resolves them at runtime, so moving a dataset directory between host
mount points does not alter prompts, predictions, or scores.

ScienceQA-IMG additionally downloads LLaVA's 23 MB evaluation archive, verifies
its SHA-256 digest, and reads `llava_test_CQM-A.json` directly from the archive.
The pinned Hugging Face mirror supplies decoded images. Materialization compares
all 4,241 rows for prompt text, answer, and image presence before selecting the
2,017 image-bearing examples, so released question IDs and prompts remain exact.

For adapter validation, the isolated Hugging Face LLaVA runtime can run the
converted official FFT checkpoint through the same reference-free generation
records. It is contextual evidence, not the controlled local LLaVA-LoRA
baseline, and is never loaded through the Inverse architecture loader.

Text-only retention is a separate, pinned lm-eval boundary. See
`docs/guides/LANGUAGE_RETENTION.md`; its raw causal-LM protocol must be identical
for Vicuna, controlled LLaVA-LoRA, and Inverse-LLaVA.

## End-to-end commands

### EvalAI authentication

Use a current token from the EvalAI profile and keep it in the ignored,
owner-readable `.env` as `EVAL_AI_AUTH_TOKEN`. Do not put the token in commands,
logs, model manifests, or submission packages. Current EvalAI JWT credentials
use `Authorization: Bearer`; older opaque tokens use `Authorization: Token`.
Check challenge participation and remaining quota before submitting. An HTTP
401 is an authentication failure, not a benchmark result. Token expiry must
be checked before scheduling an external-score handoff.

The official endpoints for this check are
`/api/participants/participant_team` and
`/api/jobs/830/remaining_submissions/` on `https://eval.ai`. Request private
test-dev submissions and retain each returned submission ID before polling.
Do not automatically repeat a submission after an ambiguous transport error.
See [EvalAI authentication settings](https://github.com/Cloud-CV/EvalAI/blob/master/settings/common.py)
and the [official CLI guide](https://cli.eval.ai/).

### Contemporary model references

The optional `predict --backend hf-multimodal` adapter uses Transformers'
`AutoProcessor` and `AutoModelForImageTextToText` with a full Hub commit SHA.
It replaces the exact declared single-turn Vicuna wrapper with the model's
native chat template, preserving question text and image position. Ambiguous
wrappers and multiple images are rejected. Image sizing follows the native
processor; its configuration hash and Transformers version enter the prediction
manifest. Scorers remain the existing benchmark scorers.

For Qwen3-VL, this follows the [official processor workflow](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl)
and [released model instructions](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
The adapter has CPU contract tests; a real-model canary and measured cost gate
are still required before a full reference evaluation. Such a model is a
contemporary reference with different training data, encoder, and resolution.
The controlled architecture comparison uses matched Vicuna/CLIP models.

```bash
invllava predict configs/benchmark/ocrbench.yaml \
  --backend hf-multimodal --model Qwen/Qwen3-VL-8B-Instruct \
  --revision "$REFERENCE_COMMIT_SHA" --dtype bfloat16 \
  --attention-backend sdpa --batch-size 1 \
  --examples /workspace/data/eval/ocrbench/examples.jsonl \
  --output /workspace/evaluations/qwen3vl/ocrbench/predictions.jsonl
```

Resolve and stage the immutable revision on an execution host first. Record
native preprocessing and greedy decoding; do not describe this as identical
processing to the CLIP-336 controlled comparison.

### Prepared benchmark runs

After freezing revisions and golden-verifying an adapter, use `fetch-eval` for
registered downloadable sources and `prepare-eval` for already acquired manual
files. The benchmark YAML is executable: its exact prompt template and declared
Vicuna-v1 conversation wrapper are rendered into every example, so the stored
prompt is exactly what the model sees. The protocol config digest is recorded
in the adjacent preparation manifest. Preparation
also hashes, reopens, fully decodes, and RGB-converts every unique image; it
writes a separate failure report and exits before inference if any image is
missing, empty, truncated, corrupt, or outside the declared image root.
Model-backed prediction, profiling, representation capture, and intervention
preparation recheck the current files before loading model weights, so damage
after preparation is also caught early. The `--allow-unverified-images` escape
hatch is diagnostic only. Generate once into an append-only store:

```bash
export INVLLAVA_ALLOW_DOWNLOADS=1
invllava predict configs/benchmark/textvqa.yaml \
  --examples /workspace/data/eval/textvqa-val.jsonl --backend native \
  --experiment configs/experiment/canonical_7b.yaml \
  --checkpoint /workspace/runs/<run>/checkpoints/<step> \
  --output /workspace/results/textvqa/predictions.jsonl --allow-download

invllava score configs/benchmark/textvqa.yaml \
  --examples /workspace/data/eval/textvqa-val.jsonl \
  --predictions /workspace/results/textvqa/predictions.jsonl \
  --output /workspace/results/textvqa/score.json
```

For the sealed Inverse-LLaVA release, use the release directory or immutable
Hub repository as `--model`; `--checkpoint` is reserved for a native training
checkpoint paired with `--experiment`:

```bash
invllava predict configs/benchmark/ocrbench.yaml \
  --examples /workspace/data/eval/ocrbench/examples.jsonl \
  --backend release --model /workspace/exports/Inverse-LLaVA-7B \
  --cache-dir /workspace/cache \
  --generation-cache kv --lora-execution unmerged --batch-size 1 \
  --output /workspace/results/ocrbench/predictions.jsonl \
  --manifest /workspace/results/ocrbench/predictions.manifest.json
```

`--cache-dir` names the shared runtime-cache root. With persistent storage,
`/workspace/cache` resolves the pinned Vicuna and CLIP snapshots from
`/workspace/cache/huggingface` and keeps the evaluation offline after staging.

Unmerged LoRA is the default and reproduces the trained computation graph.
Merged LoRA is an opt-in performance mode; its predictions and scores require a
separate parity record because low-precision weight folding can alter greedy
decoding near token ties.

The release loader applies the canonical inference numerical policy explicitly:
BF16 model weights, TF32-enabled FP32 operations, deterministic algorithms off,
and cuDNN benchmarking off. The policy is recorded in every prediction and run
manifest so native and public-loader comparisons cannot silently use different
global PyTorch defaults.

Benchmark acceptance uses batch size 1 unless a frozen benchmark record says
otherwise. Larger batches are useful throughput conditions and receive their
own prediction artifacts. Every prediction record and manifest contains the
selected batch size.

The campaign's forced flash-only SDPA diagnostic is qualified at batch size 1.
Its native mixed-length batch-4 canary stops because PyTorch flash attention
rejects the non-null padding mask. Keep that limitation distinct from ordinary
SDPA dispatch, which can select a mask-compatible kernel. A change of kernel
policy needs separately recorded qualification before comparing throughput or
reusing predictions. The current primary comparison stays at batch size 1.

`predict --backend hf-llava` runs a frozen converted LLaVA checkpoint through
the isolated Transformers reference path. Its default
`--hf-image-aspect-ratio pad` applies LLaVA-1.5's documented square padding with
the CLIP mean before the Hugging Face processor; the selected policy is recorded
in the checkpoint and generation identities. Native prediction batches images and
left-padded prompts while preserving per-sample cached-decoding parity. The
native backend also runs the controlled early-projector LLaVA experiment; the
resolved architecture, rather than a CLI alias, selects the model path.

For VQAv2 test-dev, generate the complete frozen prediction file and create a
reviewable package before submitting to the official server:

```bash
invllava package-submission configs/benchmark/vqav2_testdev.yaml \
  --examples /workspace/data/eval/vqav2-testdev.jsonl \
  --predictions /workspace/results/vqav2/predictions.jsonl \
  --full-test-questions /workspace/data/vqa/v2_OpenEnded_mscoco_test2015_questions.json \
  --output /workspace/results/vqav2/submission.json \
  --output-manifest /workspace/results/vqav2/submission.manifest.json
```

Packaging requires a complete prediction set, no references, a verified
checkpoint identity, and either a clean recorded Git commit or an execution-source
digest. The `package-vqav2` alias remains available. HF references additionally
require `--checkpoint-snapshot`, as described above. Confirm the exact test-dev
IDs and active challenge phase before submission. LLaVA evaluates 107,394
test-dev questions, normalizes answers with its M4C processor, then uploads a
447,793-row envelope with empty answers for the other 340,399 IDs. The exporter
requires every test-dev prediction and records these counts separately.
Padding outside the evaluated subset cannot establish a test-standard score.

The current EvalAI challenge 830 has test-dev phase 1793 (10 submissions/day)
and test-standard phase 1794 (1/day, 5 total). Both require the full-test
envelope. The benchmark's publication guidance recommends test-standard;
the original LLaVA comparison reports test-dev. Preserve this split distinction,
and never present a padded test-standard score as a full evaluation. Server
acceptance and account access must be confirmed before committing to full
inference. Save the phase ID, submitted bytes, status, and downloaded result.

The test-dev protocol passed full-input and pinned-converter qualification,
followed by an accepted official Inverse-LLaVA result on 11 September 2026.
Its `golden_verified` status records that protocol check, not automatic
acceptance of another checkpoint's predictions. Retain the submitted file
and the returned official result for each evaluated checkpoint.
`parse_vqav2_result` validates the returned split, four metric names, and finite
percentage values. Official aggregate scores do not provide the itemwise
information required for paired bootstrap intervals.

Sources: [LLaVA converter](https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/convert_vqav2_for_submission.py),
[VQA publication guidance](https://visualqa.org/challenge.html),
[current phase metadata](https://eval.ai/api/challenges/challenge/830/challenge_phase).

For paired evaluation-set uncertainty, use two score JSON files produced from
the same example IDs. Pass `--examples` for MME, MMBench, and MMStar. MME
resamples image groups and recomputes accuracy-plus; MMBench validates every
rotation against its scored circular group and resamples those groups; MMStar
resamples paired items within each fixed L2 capability and recomputes the
official macro average. Other protocols use paired items unless a natural
`group_id` is supplied:

```bash
invllava paired-interval \
  --left-score /workspace/results/inverse/score.json \
  --right-score /workspace/results/llava-lora/score.json \
  --resamples 10000 --seed 2026 \
  --output /workspace/results/inverse-minus-llava-ci.json
```

These intervals quantify evaluation-set uncertainty, not variation across
training seeds. With the MME protocol and `--examples`, the implementation
resamples paired image groups within category and recomputes accuracy-plus for
every bootstrap draw.

For a preregistered numeric metadata analysis, stratify the same paired score
artifacts without regenerating predictions. This is used for the TextVQA
OCR-density diagnostic:

```bash
invllava stratify-scores \
  --series Inverse /workspace/results/inverse/score.json \
  --series LLaVA-LoRA /workspace/results/llava-lora/score.json \
  --series LLaVA-FFT /workspace/results/llava-fft/score.json \
  --examples /workspace/data/eval/textvqa/examples.jsonl \
  --metadata-key ocr_token_count --boundaries 0 1 6 11 21 \
  --primary Inverse --resamples 10000 --seed 2026 \
  --output /workspace/results/textvqa-ocr-token-strata.json
```

Each nonempty stratum records its count, every model mean, and paired bootstrap
intervals for the primary model against each reference. Boundaries are lower
bounds, so this example yields `0`, `1-5`, `6-10`, `11-20`, and `21+`.
Score artifacts created under different protocol-ID schema generations remain
eligible only when benchmark, scorer, prepared-example digest, and protocol
configuration digest agree exactly. The analysis records every source protocol
ID and the content-equivalence fields in its output.

Generate aligned category figures directly from sealed score artifacts:

```bash
invllava plot-score-breakdown \
  --series Inverse /workspace/results/inverse/score.json \
  --series LLaVA-LoRA /workspace/results/llava-lora/score.json \
  --series LLaVA-FFT /workspace/results/llava-fft/score.json \
  --detail-key category_accuracy --title OCRBench --ylabel Accuracy \
  --output /workspace/results/ocrbench-categories.pdf
```

The command requires one benchmark/protocol and identical category keys. Its
JSON sidecar records source hashes, checkpoint identities, plotted values, and
the figure hash.
