# Inverse-LLaVA: Eliminating Alignment Pre-training Through Text-to-Vision Mapping

**Project Website:** [https://inverse-llava.github.io](https://inverse-llava.github.io)

### Authors

* [Xuhui Zhan](https://xuhuizhan5.github.io) (Data Science Institute, Vanderbilt University)
* [Tyler Derr](https://tylersnetwork.github.io) (Computer Science Department, Vanderbilt University)

### Abstract

Traditional multimodal learning approaches require expensive alignment pre-training to bridge vision and language modalities, typically projecting visual features into discrete text token spaces. We challenge both fundamental assumptions underlying this paradigm by proposing **Inverse-LLaVA**, a novel approach that eliminates alignment pre-training entirely while inverting the conventional mapping direction. Rather than projecting visual features to text space, our method maps text embeddings into continuous visual representation space and performs fusion within transformer intermediate layers. Through selective additive components in attention mechanisms, we enable dynamic integration of visual and textual representations without requiring massive image-text alignment datasets. Comprehensive experiments across nine multimodal benchmarks demonstrate nuanced performance trade-offs: Inverse-LLaVA achieves notable improvements on reasoning-intensive and cognitive tasks (MM-VET: +0.2%, VizWiz: +1.8%, ScienceQA: +0.2%, cognitive reasoning: +27.2%), while showing expected decreases in perception tasks requiring memorized visual-text associations (celebrity recognition: -49.5%, OCR: -21.3%). These results provide the first empirical evidence that alignment pre-training is not necessary for effective multimodal learning, particularly for complex reasoning tasks. Our work establishes the feasibility of a new paradigm that reduces computational requirements by 45%, challenges conventional wisdom about modality fusion, and opens new research directions for efficient multimodal architectures that preserve modality-specific characteristics. Our project website with code and additional resources is available at [https://inverse-llava.github.io](https://inverse-llava.github.io).

---

This codebase is adapted from the original [LLaVA](https://github.com/haotian-liu/LLaVA) project. We have made modifications to implement Inverse-LLaVA and Inverse-LLaVA-HD.

## Training

To reproduce the training for our models, please use the following scripts:

* **Inverse-LLaVA:**
  ```bash
  bash llava/scripts/v1_5/fusion_finetune_lora.sh
  ```
* **Inverse-LLaVA-HD:**
  ```bash
  bash llava/scripts/v1_5/fusion_finetune_lora_HD.sh
  ```

## Evaluation

The evaluation process is identical to the one provided in the original LLaVA documentation: [LLaVA Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

However, there are some specific settings for Inverse-LLaVA to be aware of:

* `--use_mm_proj False`: This is a critical setting for Inverse-LLaVA. It disables the projection layer for the vision encoder output, which is a core aspect of our approach.
* `--pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin`: This line has no effect in our model because `--use_mm_proj` is set to `False`.
* `--mm_vision_select_layer`: This parameter determines which hidden states from the vision encoder are used.
  * For **Inverse-LLaVA**, this is set to `"-1"`, meaning only the last layer's hidden state is used.
  * For **Inverse-LLaVA-HD**, this is set to `"-1,-2"`, meaning the last and second-to-last layers' hidden states are used.

## Inverse-LLaVA Specific Hyperparameters

The following hyperparameters are specific to Inverse-LLaVA and control the fusion mechanism:

```
--model_type fusion_llama
--use_vision_fusion True
--stable_fusion False
--mm_hidden_size 1024
--fusion_alpha 1.0
--fusion_hidden_dim 128
--fusion_dropout 0.1
--fusion_targets 'q,k,v'
--lora_layer_ids_to_skip '0'
--layer_ids_to_inject '0'
```

* `--model_type fusion_llama`: Specifies the model architecture to use our fusion mechanism with LLaMA.
* `--use_vision_fusion True`: Enables the text-to-vision fusion within the transformer layers.
* `--stable_fusion False`: A flag for a specific fusion variant. `False` is the default for Inverse-LLaVA.
* `--mm_hidden_size 1024`: The hidden size of the visual features. For the HD version, this is 2048.
* `--fusion_alpha 1.0`: A weighting parameter for the fusion process.
* `--fusion_hidden_dim 128`: The hidden dimension of the fusion layer.
* `--fusion_dropout 0.1`: The dropout rate for the fusion layer.
* `--fusion_targets 'q,k,v'`:  Specifies that the fusion should be applied to the query, key, and value projections in the attention mechanism.
* `--lora_layer_ids_to_skip '0'`:  Specifies which LoRA layers to skip.
* `--layer_ids_to_inject '0'`: Specifies which layers to inject the fusion into.

## Citation

If you find Inverse-LLaVA useful for your research and applications, please cite using this BibTeX:

```
@article{zhan2025inversellava,
  title={Inverse-LLaVA: Eliminating Alignment Pre-training Through Text-to-Vision Mapping},
  author={Zhan, Xuhui and Derr, Tyler},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```
