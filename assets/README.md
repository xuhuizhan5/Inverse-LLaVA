# Artwork and example provenance

`inverse-llava-logo.svg` is the horizontal wordmark; `inverse-llava-mark.svg`
is the standalone emblem. The left-facing llama and returning blue fold are
original artwork, developed from AI-assisted concept studies and redrawn as
simple vector paths. No other project's logo was used as a reference.

Both files have transparent backgrounds and adapt to light and dark color
schemes. Use the full logo at 320 pixels wide or larger and the emblem at
24 pixels or larger. Preserve the built-in clear space and proportions.
The SVGs contain their own geometry, with no embedded bitmap, external font,
script or network dependency. Their accessible titles identify the project.

The wordmark uses outlined [Inter 4.1](https://github.com/rsms/inter/tree/v4.1)
at weight 620; the descriptor uses weight 450. Inter is by Rasmus Andersson and
the Inter Project Authors, under the
[SIL Open Font License 1.1](https://github.com/rsms/inter/blob/v4.1/LICENSE.txt).
No font binary is distributed here. The navy (`#142E40`) and blue (`#0072B2`)
coordinate with the scientific figures; dark backgrounds use `#E6EDF3` and
`#56B4E9` for contrast.

`overview.svg` is the introductory method illustration. It contains one shared
image-question input and the recorded answers from Inverse-LLaVA and the official
LLaVA-1.5 LoRA/FFT references. `architecture.svg` gives the detailed fusion path.
The diagrams use editable SVG shapes and text; only the benchmark image is raster.

## Selected example

The free-form question is **VizWiz, `VizWiz_test_00006540.jpg`**:
“What color is it?” The original photograph shows a white shoe.
Inverse-LLaVA answers **“White”**; both official LLaVA references answer
**“Blue”**. Their recorded consensus item scores are 1, 0, and 0.
The question and answers are verbatim, and the photograph is embedded without
cropping or alteration. The fixed chat wrapper and response instructions are
omitted from the display only; inference used the full saved prompt.
The figure labels the reference as "Reference: White (1)" and includes the
VizWiz sample ID. Here, 1 denotes full consensus credit, not the number of
annotators: nine annotations say "white" and one says "shoe white." Answer text stays
neutral; red "Both wrong" and green "Correct" identify the scored outcomes,
with explicit words and numerical scores preserving the meaning without color.

`overview.json` records the complete prompt, reference, unedited answers,
checkpoint identities and evidence hashes, checked against the accepted primary
evaluation. This visually inspected example was selected for an unambiguous,
compact illustration from the primary-only successes. It is distinct from
every example in the existing qualitative casebooks.
It does not estimate how frequently either model succeeds.

Image and question credit: [VizWiz-VQA](https://vizwiz.org/tasks-and-datasets/vqa/).
The dataset page licenses this work under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The image is reproduced unchanged. Project artwork does not change these
third-party image rights.
