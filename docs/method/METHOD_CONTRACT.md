# Inverse-LLaVA method contract

This document specifies the architecture, tensor shapes, initialization, and
runtime invariants used by the manuscript's reference configuration.

## Tensor path

Let `B` be batch size, `S` the expanded text/patch sequence length, `d_h` the
language hidden size, and `d_v` the selected CLIP feature size.

- The initially expanded `H ∈ R[B,S,d_h]` contains token embeddings at text
  positions and zero vectors at newly inserted patch positions. At a deeper
  fusion ablation, ordinary preceding layers may contextualize both positions.
- `V ∈ R[B,S,d_v]` contains CLIP patch features at patch positions and zero
  vectors at text positions.
- `W_t2v^P: d_h → d_v` maps the intermediate language state into visual space
  separately for each `P ∈ {Q,K,V}` branch.
- `N_P` is a target-specific RMS normalization of native visual features with
  epsilon `1e-6`.
- For `P ∈ {Q,K,V}`, `W_concat^P: 2d_v → d_P` maps the concatenation back to
  the attention-projection output size.
- `alpha_P` is target-specific and either learned or fixed by an ablation.

At a selected layer:

`P = W_P H + alpha_P W_concat^P concat(mask_text(W_t2v^P H), mask_vision(N_P(V)))`.

Concatenation is on the last dimension. Canonical Vicuna-7B has `d_h=4096`,
`d_v=1024`, 32 layers, and fusion in layer 0 for Q, K, and V. CLIP ViT-L/14@336
contributes 576 patch tokens from its final hidden layer. The HD setting
concatenates the final and penultimate CLIP layers, so `d_v=2048`.

`d_v` is fixed by the selected native visual features. Changing the mapper
output alone would prevent fusion with `V`; reducing both streams would
introduce an additional visual projection. Mapper-capacity ablations therefore
factor `W_t2v:4096→1024` through rank 256 or 512 and retain the 1,024-D output.

## Behavioral invariants

- The number of `<image>` placeholders must exactly equal the number of image
  feature tensors for each sample.
- Image placeholders are removed before language embedding lookup and replaced
  by the corresponding patch count. Patch labels are `-100`.
- Padding positions belong to neither modality.
- Fusion state is an explicit forward argument. A module must never retain image
  features between requests.
- Initial prompt decoding fuses the full expanded sequence. The portable cached
  path then fuses each new text token with a zero visual branch and reuses KV
  states. The primary evaluations use BF16, SDPA, and KV caching. FP32
  full-sequence recomputation is available as an explicitly selected diagnostic
  runtime; results from different precision or cache policies remain separate.
- Text-only evaluation passes a zero visual branch and therefore measures the
  language checkpoint/adaptation without fabricating an image.
- The CLIP tower is frozen in the canonical experiment. Layer-0 ordinary Q/K/V
  projections retain their loaded language weights; the fusion path is added.
- Canonical LoRA targets transformer attention and MLP linears outside the first
  fusion layer (Q/K/V/O and gate/up/down). Rank is 128, alpha 256, and dropout
  is 0.05.
- Training batches use deterministic modality/length grouping, as in the recorded
  launch recipe, to reduce padding without changing the selected sample set.
- In controlled-LLaVA text-only batches, the early projector is numerically
  bypassed but retains a zero autograd dependency so ordinary DDP remains valid.

## Implementation details

Q, K, and V have distinct learned `4096×1024` text-to-vision matrices,
1024-wide visual RMS-normalization weights, and learned scales. The equations
use target-specific indices. The primary comparison uses the verified
retrained checkpoint at update 5198.

Fresh initialization uses scale 1.0, mapper standard deviation `1/sqrt(d_v)`,
and concatenation-output standard deviation `1e-4`.
Fresh fusion and LoRA tensors use FP32 initialization
arithmetic and are cast to BF16 before training. The canonical forward uses
BF16 model tensors, DeepSpeed keeps FP32 optimizer/master state, and TF32 is
permitted only for eligible FP32 CUDA operations. “Full fine-tuning” names a
trainable scope and does not imply an FP32 forward. Release conversion copies
every learned fusion and LoRA tensor exactly.

Canonical Vicuna attention, MLP, and language-head linears are bias-free. The
wrapped ordinary Q/K/V branch inherits that property. Fusion contains the
text-to-vision and concatenation-output matrices, visual normalization weights,
and a scale for each target, with no additive bias. LoRA also uses no bias.
Adding fusion biases defines a separate ablation. Equation (2) omits a
text-to-vision bias, and Algorithm 1
omits a language-head bias because the pinned Vicuna head has none.

The early-projector baseline uses a two-layer GELU MLP with biases. Its
initialization is explicit in the experiment configuration:

- The matched single-stage control uses `initialization: random`, the official
  projector constructor's seeded `nn.Linear` initialization, and no alignment
  checkpoint. It receives the same instruction subset and LoRA recipe as the
  corresponding Inverse-LLaVA condition.
- A stage-two LLaVA recipe uses `initialization: checkpoint` and requires the
  converted alignment projector with its pinned hash.
- Official LoRA and full-fine-tuned checkpoints retain their learned projector
  weights and serve as pretrained reference models.

The trainer rejects an alignment checkpoint for a random-projector experiment
and rejects a missing or mismatched checkpoint for a checkpoint-initialized
experiment. The feature-interface experiment matrix declares the matched
CLIP-layer controls; retain the initialization audit with each run.

## Complexity

For one concat-fusion layer and target output sizes `d_Q,d_K,d_V`, parameters are

`3 d_h d_v + 2 d_v (d_Q+d_K+d_V) + 3 d_v + 3 learned scales`.

The corresponding multiply-accumulates at expanded length `S` are the weight
terms multiplied by `S`. Report this as calculated, not measured. Activation
memory depends on implementation lifetime and precision and must be profiled.

## Validation and result provenance

The manuscript specifies the method. Checkpoint conversion verifies tensor
inventory, shapes, duplicate-file agreement, matrix orientation, hashes, and
the base-projection identity against pinned Vicuna. Runtime checks cover
deterministic loading and generation, official prompt and scorer fixtures,
and image decoding. The main table uses the verified retraining and the
official reference checkpoints under the stated evaluation protocols.
Hardware-specific timings retain their measured environment and scope.
