# Inverse-LLaVA: Rethinking Multimodal Alignment via Text-to-Vision Mapping

**Project Website:** [https://inverse-llava.github.io](https://inverse-llava.github.io)

### Authors

* [Xuhui Zhan](https://xuhuizhan5.github.io) (Data Science Institute, Vanderbilt University)
* [Tyler Derr](https://tylersnetwork.github.io) (Data Science Institute and Computer Science Department, Vanderbilt University)

### Abstract

Traditional multimodal learning approaches rely on alignment pre-training to bridge vision and language modalities, typically by projecting visual features into discrete text token spaces using large-scale image-text data. We revisit this design choice and propose **Inverse-LLaVA**, a multimodal architecture that inverts the conventional mapping direction by projecting text embeddings into continuous visual representation space and performing fusion within intermediate transformer layers. This representation-first design enables effective multimodal reasoning without relying on an explicit alignment pretraining stage and significantly reduces dependence on large alignment datasets. Across nine multimodal benchmarks, Inverse-LLaVA demonstrates strong learning efficiency under reduced supervision, achieving substantial gains on reasoning-intensive tasks while exhibiting selective performance drops on perception tasks that depend on explicit visual-text grounding. Our analysis indicates that these trade-offs primarily reflect differences in supervision regime rather than architectural limitations. Together, these results show that alignment pretraining is not strictly required for effective multimodal reasoning and highlight the importance of preserving continuous modality representations, opening a new direction for multimodal architecture design that decouples representation structure from supervision regime for more flexible and efficient multimodal systems.

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

For an apples-to-apples comparison with LLaVA-1.5, we use the same instruction-tuning dataset, backbone models, and optimization settings. The dataset preparation and structure are identical. You can find more details in the [original LLaVA repository](https://github.com/haotian-liu/LLaVA). Our approach is most closely aligned with its LoRA training methodology. The experiments were conducted on 8 NVIDIA A100 GPUs, consistent with the LLaVA-1.5 training setup.

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
@article{zhan2025inverse,
  title={Inverse-LLaVA: Rethinking Multimodal Alignment via Text-to-Vision Mapping},
  author={Zhan, Xuhui and Derr, Tyler},
  journal={arXiv preprint arXiv:2508.12466},
  year={2025}
}
```
