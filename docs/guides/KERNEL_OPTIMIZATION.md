# Optional kernel optimization

The portable runtime remains the reference path. It uses PyTorch scaled-dot-product
attention (`sdpa`), allowing PyTorch to select a compatible fused CUDA attention
kernel without adding a FlashAttention fork. `cuda_1gpu_inductor` is a separate,
optional execution runtime that compiles only the inverse-fusion modules with
TorchInductor and its CUDA/Triton code generation.

This option does not alter model weights, data, precision, prompts, or benchmark
protocols. Runtime configuration is excluded from the scientific content ID, but
the complete resolved runtime, PyTorch/Triton versions, GPU, compiler mode, and
checkpoint hashes remain in run artifacts. Exact resume requires the same resolved
runtime; never switch compilation on or off inside a run.

## Why this boundary

- `sdpa` is the portable default and `eager` remains the debugging reference.
- Compilation is requested only with `--runtime-ref cuda_1gpu_inductor`.
- An unavailable CUDA, Triton, or compiler path is an error—there is no silent
  fallback that could contaminate a timing comparison.
- `nn.Module.compile` is applied in place and the implementation verifies that
  checkpoint state-dictionary keys do not change.
- Compiler and Triton caches live under the declared runtime cache root.
- Representation capture deliberately requires the uncompiled runtime because
  compiler graph transformations can invalidate module-hook interpretations.

The supplied candidate uses the conservative default compiler mode, static
shapes, and fusion-only scope. Whole-language-model `max-autotune` is not a
supported recipe: repeated validation trials produced different greedy-token parity
outcomes across fresh and warm compiler caches even though logits and gradients
met BF16 tolerances. Fusion-only compilation must pass validation on the target GPU;
it is not a claimed universal optimum.

## Qualification ladder

First run the no-download audit:

```bash
invllava kernel-audit configs/runtime/cuda_1gpu_inductor.yaml \
  --device cuda --dtype bfloat16 \
  --output /workspace/runs/kernel-audit.json
```

The audit creates identical tiny Inverse-LLaVA models and requires:

1. unchanged checkpoint keys;
2. BF16 forward logits/loss within the recorded tolerance;
3. fusion gradients within the same tolerance;
4. exact deterministic greedy tokens;
5. measured cold-start compile cost and warmed median/IQR timing.

The tiny model catches integration and numerical failures; its speedup is **not**
a 7B or H100 throughput estimate. Before using the runtime for paper evidence:

1. Compare portable and compiled inference from the same real checkpoint on a
   frozen calibration panel; require exact prompts, sample coverage, and scored
   outputs, and inspect any changed greedy answer.
2. Profile both paths with identical batch/sequence/decode work after compilation
   and cache warm-up; report compile latency separately.
3. Run the 128-example training contract from the same initialization and seed,
   compare loss/gradient traces, checkpoint inventory, exact resume, and the frozen
   evaluation panel.
4. Run a 5% paired calibration only if compiled training passes the smaller gate.

If inference passes but training diverges materially, use compilation for
evaluation only. If deterministic answers or benchmark scores change, the portable
runtime remains authoritative. A speedup never overrides a failed parity gate.

## Real-model commands

Profile an already trained checkpoint without changing it:

```bash
INVLLAVA_ALLOW_DOWNLOADS=1 invllava profile-native \
  configs/experiment/canonical_7b.yaml \
  --runtime-ref cuda_1gpu_inductor \
  --checkpoint /workspace/runs/inverse-llava-7b-seed42/checkpoints/step-0005198 \
  --examples /workspace/data/eval/profile-panel.jsonl \
  --batch-sizes 1 4 --decode-tokens 32 \
  --warmups 10 --repetitions 30 \
  --output /workspace/runs/profiles/inverse-inductor.json \
  --allow-download
```

After the training gates pass, the same runtime override can be supplied to
`scripts/launch_train.py`. Keep portable and optimized runs in different run
roots and compare checkpoints only from matched scientific IDs.

## Profiling tools

Use the lowest-overhead tool that answers the question:

- the built-in synchronized profiler for paper latency, memory, and throughput;
- `kernel-audit --trace <new-file.json>` for a small PyTorch Profiler trace;
- NVIDIA Nsight Systems for CPU/GPU scheduling, kernel-launch gaps, data stalls,
  and compiler-generated kernel names;
- NVIDIA Nsight Compute only for a short, already-identified hot kernel because
  replay and metric collection are expensive.

`tools/profiling/nsys_profile.sh` is an optional wrapper and refuses overwrite.
Wrap a short `profile-native` invocation rather than a full training run. Nsight
availability is not a repository dependency.

Do not add custom Triton kernels until these profiles show that a stable fusion
operation is a material hotspot. The first external-kernel candidate, if the
profile supports it, is Liger fused linear cross entropy: the current training
forward materializes full vocabulary logits before computing loss, which can be
a larger memory target than the small fusion block. Qualify it through an
optional runtime with forward-loss, gradient, update, resume, and frozen-panel
comparisons. RMSNorm, RoPE, and SwiGLU fusion follow only when their measured
share is material. FP8, quantization, fused optimizers, and altered attention
implementations require a separate scientific study.

## Attention and decoding candidates

PyTorch SDPA is also the fused-attention integration boundary. On a supported
GPU it selects an eligible CUDA backend; the profiler trace and environment
record must identify the backend actually used. H100/H200 qualification first
measures SDPA with the CUDA stack already present in the publication image.
Transformer Engine or a separately installed FlashAttention implementation is
considered only after profiling shows attention is material and exact logit,
generation, score, gradient, and memory comparisons pass. The optional FP32
full-recompute diagnostic remains on the portable path because fused attention
backends primarily target BF16/FP16 execution.

DFlash2 is a speculative-decoding serving method that requires a compatible
draft checkpoint. There is currently no frozen DFlash draft for this custom
multimodal Vicuna model, and the reference evaluations use short, batch-one
benchmark answers. It is therefore outside the training and scientific
evaluation path. A later serving study would need its own draft training,
checkpoint identity, acceptance statistics, and exact target-model output
validation.
Liger follows the same measured-candidate rule for RMSNorm, RoPE, SwiGLU, and
loss kernels: profile first, integrate through an optional runtime, and retain
portable loading for the core model graph.

## Primary references

- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [PyTorch compiler caching](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [PyTorch scaled-dot-product attention](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [NVIDIA Transformer Engine attention backends](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html)
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- [SGLang speculative-decoding documentation](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/speculative_decoding.mdx)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
