# Third-party notices

## Source code

Inverse-LLaVA source code is distributed under [Apache-2.0](LICENSE), following
the code licenses of [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)
and [FastChat](https://github.com/lm-sys/FastChat/blob/main/LICENSE).
[NOTICE](NOTICE) records acknowledgments and attribution.

The VQA normalization tables and TextVQA processing conventions used in
`src/invllava/eval/protocols/evalai.py` derive from the official VQA evaluator
and Pythia/MMF through LLaVA's `m4c_evaluator.py`. The Pythia
[BSD license](third_party/licenses/Pythia-BSD.txt) is retained. The local
implementation separates benchmark-specific normalization paths; its tests
compare against pinned reference implementations.

Installed libraries and externally downloaded scorers retain their own
licenses. [Source identities](third_party/LOCKS.md) identify the dependency
locks, model revisions, data sources, and evaluator references used here.

## Models and datasets

The code license does not license model weights or training data. In particular,
[Vicuna v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) is based on Llama 2
and uses the [Llama 2 Community License](https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/LICENSE.txt),
including its applicable use and redistribution requirements. These terms must
be considered when distributing trained adapters, fusion weights, or merged
models. A model release needs its own model card and applicable license files;
the Apache-2.0 declaration for this repository does not replace them.

[CLIP](https://github.com/openai/CLIP/blob/main/LICENSE) uses the MIT license.
The LLaVA instruction mixture combines data from several providers; annotations
and images retain their respective source terms. Download benchmark data from
the sources in the dataset and benchmark configurations and follow each
provider's conditions. Bulk weights and datasets are not bundled here.

## Artwork

The llama mark and method diagrams are project artwork. The outlined Inter
wordmark and the VizWiz example image have separate attribution in
[assets/README.md](assets/README.md). The VizWiz image and question are credited
under CC BY 4.0. Project artwork does not change third-party image or font rights.
