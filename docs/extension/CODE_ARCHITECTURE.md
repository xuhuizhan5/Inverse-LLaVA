# Code architecture

The package separates scientific configuration from execution and benchmark
adapters. Dependency flow stays one-way:

```text
strict YAML configs ──► data/model/eval services ──► train or inference runtime
                              │                              │
                              └────────► immutable artifacts ◄┘
```

## Stable boundaries

- `config/` owns strict schemas, recipe resolution, content IDs, and atomic
  experiment-matrix materialization. Unknown settings fail before execution.
- `data/` owns provider acquisition, archive safety, canonical layout assembly,
  annotation normalization, image decoding, and prepared-data manifests.
- `model/` owns the manuscript-shaped tensor path. Fusion state is an explicit
  request value; no image state is stored on a module between forwards.
- `train/` owns deterministic sampling, metrics, checkpoint state, exact resume,
  and optional tracking. It is independent of the compute provider.
- `eval/` owns frozen prompts, dataset adapters, append-only predictions,
  coverage checks, scoring, and external-submission packaging.
- `runtime/` composes frozen native and Hugging Face references for execution.
  Loading owns revision checks, device/dtype placement, cache policy, and model
  identity. Prediction and analysis consume these runtimes instead of rebuilding
  upstream models independently.
- `release/` is the small public checkpoint interface. A local directory and a
  Hub repository resolve to the same six-file bundle and `load_pretrained`
  path. PEFT/fusion checkpoint conversion is an explicit import operation and
  never enters ordinary inference.
- `analysis/` performs read-only model passes for representation capture and
  profiling, then consumes immutable predictions, scores, and feature arrays.
  It never updates model parameters or replaces recorded benchmark inference.
- `artifacts/` supplies atomic writes, hashing, run manifests, and scoped
  cleanup shared across all workflows.

Host-specific launchers can select resources and mount caches while calling
the package's portable interfaces. They must not redefine scientific settings.

## Extension rules

Add a new scientific knob to a strict schema and a reviewed YAML recipe before
using it in model code. Add a benchmark through the protocol and dataset
registries described in [Adding a benchmark](ADDING_A_BENCHMARK.md). Put
provider-specific downloads behind explicit authorization and bind every
accepted source to a revision and digest. Keep generated results out of source
modules; use manifests and ledgers instead.

Prefer a small composition function over copying model construction into a new
command. Prefer an explicit dataclass or Pydantic model over an untyped mapping
at module boundaries. Test pure invariants locally, full dependency contracts
in the pinned container, and real checkpoints in a bounded canary before long
experiments.

The public Python surface is intentionally small. Command-line workflows are
the reproducibility interface; internal modules remain available for research
extensions but are not promised as a versioned library API.

## Shared contracts and backend boundaries

| Workflow | Shared contract | Backend-specific boundary |
|---|---|---|
| benchmark inference | frozen examples, prompt hash, generation record, prediction manifest | native, release, or verified HF LLaVA loader |
| representation capture | ordered sample IDs, `hidden.last.<layer>`, pooling metadata, NPZ/JSON hash pair | separate-stream fusion features, inserted-token LLaVA features, or text-only causal states |
| qualitative study | arbitrary labeled prediction/score series and disjoint deterministic correctness buckets | each model retains its original output and checkpoint identity |
| language retention | one pinned lm-eval suite and raw causal-LM protocol | native, HF causal, or HF LLaVA language path |
| performance profile | fixed prompt panel, synchronized warmups/repetitions, equal decode work | native controlled pair for primary timing; provider runtimes remain contextual |

Model loading is independent of benchmark and training-data preparation when a
sealed release is available. A training `ResolvedExperiment` remains the right
input for training and native research checkpoints because it binds model,
data, optimizer, and runtime into one scientific identity. Release analysis
uses the release model specification directly; it does not inherit unrelated
training-data validation.

The common analysis surface stops at quantities with the same semantics.
Terminal prompt states can be compared across architectures. Visual-token means
cannot be treated as equivalent when one model inserts projected patches and
another keeps a separate visual branch, so those arrays retain explicit names
and are analyzed within their valid spaces.

## Design trade-offs

- Strict schemas and checksums reject ambiguous artifacts early, at the cost of
  requiring immutable revisions and complete metadata.
- Delta checkpoints reduce storage and preserve upstream license boundaries,
  at the cost of resolving Vicuna and CLIP once per shared cache.
- The native Llama core makes fusion, masks, and profiling inspectable, at the
  cost of qualifying supported Transformers/PyTorch ranges against the upstream
  reference implementation.
- Optional compilation is selected by runtime config. Portable SDPA remains
  available on every supported machine and owns the reference outputs.
