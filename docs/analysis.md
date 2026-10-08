# Analysis

All plots should be traceable to saved measurements, checkpoint identities and
sample IDs. Rendering changes do not require rerunning inference.

The [research artifacts](https://huggingface.co/xuhuizhan5/Inverse-LLaVA-research-artifacts)
include saved training histories, numerical study summaries and the matched
100-pair representation tensors. Their metadata preserve sample order,
feature locations, pooling and checkpoint hashes. The collection's guide
explains selective downloading; benchmark images remain with their providers.

## Training curves

`invllava plot-training-curves --help` describes the recorded-metrics interface.
Compare loss curves only when data selection, target masking and loss reduction
are matched. Label the horizontal axis as optimizer updates, processed examples
or supervised tokens, according to the quantity actually plotted. Lower
training loss alone does not establish higher benchmark performance.

## Efficiency

Follow [Profiling](guides/PROFILING.md). Fix image resolution, prompts, batch size,
output length, precision, attention backend and LoRA merge policy. Warm up,
synchronize devices and report distributions over repeated measurements.
Separate image preparation, prefill and decoding. Report hardware alongside
latency and memory; keep analytical FLOPs and parameter counts distinct.

## Representations and examples

Follow [Representations and cases](guides/REPRESENTATIONS_AND_CASES.md). Capture
the same image-question pairs and named feature locations in each model.
Use quantitative high-dimensional measures alongside PCA/t-SNE. Joint modality
plots need a common feature space within each fit; independently fitted panels
do not share coordinate axes. A 2D overlap alone is not a performance measure.

Qualitative figures should add different examples and task types. Save complete
prompts, original images, references and all compared outputs. Identify any
display abbreviation, OCR hints or outcome-based selection. Include successes
and failures in the case study; avoid reusing one example throughout the paper.

## Figure style

Use `invllava.analysis.plot_style` for fonts and the colorblind-accessible
palette: blue for Inverse-LLaVA, orange for LLaVA-LoRA, green for LLaVA-FFT.
Keep these model identities consistent. Direction-of-change plots can use
neutral markers with a zero reference and confidence intervals, avoiding a
second conflicting red/green meaning. Bars start at zero; axes carry units.
Use horizontal labels and vector PDF/SVG output whenever possible.
