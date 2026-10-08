# Architecture ablation definitions

All variants keep the selected data IDs, epoch/update policy, seed, backbone,
image resolution, prompt, and local evaluation panel fixed unless stated.
The [component matrix](../../configs/experiment/component_ablation_matrix.yaml)
defines the matched 5% comparisons. The
[extended matrix](../../configs/experiment/ablation_matrix.yaml) supplies
additional scale, depth, and capacity settings.

- **Mapper frozen:** initialize `W_t2v` identically to canonical and freeze it;
  concat output projections and scales remain trainable.
- **Fusion disabled, matched LoRA targeting:** remove the additive branch while
  keeping layer 0 excluded from LoRA, exactly as in canonical. This is the clean
  branch-removal comparison, although it necessarily has fewer trainable
  parameters because the removed branch is not replaced.
- **Fusion disabled, all-layer LoRA:** remove the branch and restore ordinary
  LoRA in layer 0. This secondary capacity-compensation control is reported
  separately and is not described as a one-factor ablation.
- **Addition:** replace `[T;V]` by `T+V`, then apply target-specific `d_v→d_P`
  projections. Report its smaller parameter count.
- **Gate:** `g=sigmoid(W_g[T;V])`, `F=gT+(1-g)V`, followed by target-specific
  projections. One shared gate is used across the Q/K/V branches. `W_g` and its
  bias start at zero, so the initial blend is 0.5/0.5. This is the exact gating
  design being tested; conclusions do not generalize to all gates.
- **Scale:** fix each target scale to 0.25, 0.5, or 1.0 versus the canonical
  learned scales, and report learned final values.
- **Depth:** insert the otherwise identical branch at layers 0, 4, 8, 16, or 24
  of 32, excluding LoRA only at the selected layer. Layer 16 is the primary
  nonzero-depth condition: it balances a contextualized text state against 16
  remaining layers of multimodal integration. SmolVLA's use of the first half
  of its VLM layers motivates testing the midpoint, while its different action
  task and layer-truncation design do not establish that midpoint fusion is
  inherently superior. Multi-layer fusion is promoted only if the single
  midpoint condition is informative.
- **Mapper rank:** factor each `W_t2v:4096→1024` through an internal rank of 256
  or 512 while keeping its output in the native 1,024-D CLIP feature space.
  This is the clean sensitivity test for mapper capacity. The three direct maps
  contain 12,582,912 weights; rank 512 contains 7,864,320 and rank 256 contains
  3,932,160. The factorized initialization is variance-normalized across ranks.
- **Vision layer:** select the penultimate rather than final CLIP hidden layer
  while keeping the encoder and 1,024-D fusion unchanged. This checks whether a
  reported difference could reflect the feature-layer choice used by LLaVA.
- **Native 2048-D/HD:** concatenate the final two CLIP layers as in Inverse-LLaVA-HD.
  This changes both width and visual layer content; interpret it as a joint
  feature-width and layer-content comparison.

The mapper output dimension is fixed by the selected visual encoder features:
1,024 for one CLIP ViT-L/14 hidden layer and 2,048 for the HD feature
concatenation. Reducing that external dimension would require an additional
visual projection and would change the reference architecture. It is
therefore excluded from the primary ablation matrix.

Depth is interpreted as an integration trade-off. Layer 0 exposes visual
information to the full decoder but maps a minimally contextualized text state.
Layer 16 maps a richer intermediate text state and leaves half the decoder for
cross-modal reasoning. Layer 24 leaves only eight layers for integration. The
layerwise representation study measures this hypothesis; benchmark scores
decide it.

Reference: [SmolVLA technical report, Sections 3 and 4.3](https://arxiv.org/html/2506.01844),
which truncates the upper VLM layers and reports its own task-specific depth
ablation.

Loss/gradient curves screen implementation collapse. Component claims use the
frozen balanced evaluation panel at 5%, then selected confirmation at 20%.
The preregistered selection score first maps each official score to its fraction
of the benchmark maximum, averages ScienceQA and AI2D into one science/reasoning
domain, and then gives equal weight to science/reasoning, MMStar, MME perception,
MME cognition, and OCRBench. Raw scores with incompatible ranges are never
averaged directly.
