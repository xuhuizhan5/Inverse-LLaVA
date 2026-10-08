"""Shared publication colors; hue identifies a series, never its rank.

Model comparisons, modality views, and intervention/control plots have separate
legends. Heatmaps use a sequential scale because they encode numeric magnitude.
Markers and explicit labels keep these meanings available without color.
"""

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"
CATEGORICAL = (BLUE, ORANGE, GREEN)
MODEL_COLORS = dict(zip(("Inverse-LLaVA", "LLaVA-LoRA", "LLaVA-FFT"), CATEGORICAL, strict=True))
MODALITY_COLORS = {"Text": ORANGE, "Vision": BLUE}
PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
