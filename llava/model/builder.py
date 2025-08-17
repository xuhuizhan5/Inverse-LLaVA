#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_FUSION_TARGETS, DEFAULT_FUSION_HIDDEN_DIM, DEFAULT_FUSION_ALPHA, DEFAULT_FUSION_DROPOUT, DEFAULT_FUSION_VISION_DIM, DEFAULT_FUSION_LAYER_IDS_TO_INJECT
from llava.model.get_fusion_model import get_fusion_model

def print_up_A_parameters(model):
    """Print all parameter names containing 'up_A'"""
    print("="*50)
    print("Parameters containing 'up_A':")
    count = 0
    for name, param in model.named_parameters():
        if "up_A" in name:
            print(f"- {name}: {param.shape}, dtype: {param.dtype}, device: {param.device}")
            count += 1
    print(f"Total parameters containing 'up_A': {count}")
    print("="*50)

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        # kwargs['torch_dtype'] = torch.float16
        # kwargs['torch_dtype'] = torch.bfloat16
        kwargs['torch_dtype'] = torch.float32
        

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'
                
    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            name = "Fusion" if 'fusion' in model_name.lower() else "LLaVA"
            if 'fusion' in model_name.lower():
                from llava.model.language_model.fusion_llama import FusionLlamaConfig
                lora_cfg_pretrained = FusionLlamaConfig.from_pretrained(model_path)
                # lora_cfg_pretrained.torch_dtype = torch.bfloat16
                lora_cfg_pretrained.use_cache = False
                print("="*100)
                print(f"lora_cfg_pretrained: {lora_cfg_pretrained}")
                print("="*100)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print('Loading Fusion model from base model...')
                model = FusionLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
                
                # Print parameters before adding fusion components
                print("BEFORE ADDING FUSION COMPONENTS:")
                print_up_A_parameters(model)
                
                print('Adding fusion components and initializing fusion layers with original weights...')
                # Then add fusion layers AFTER weights are loaded
                if getattr(model.config, "use_vision_fusion", True):
                    model = get_fusion_model(
                        model,
                        vision_dim=getattr(model.config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
                        fusion_targets=getattr(model.config, "fusion_targets", DEFAULT_FUSION_TARGETS),
                        alpha=getattr(model.config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
                        dropout=getattr(model.config, "fusion_dropout", DEFAULT_FUSION_DROPOUT),
                        stable_fusion=getattr(model.config, "stable_fusion", False),
                        fusion_hidden_dim=getattr(model.config, "fusion_hidden_dim", DEFAULT_FUSION_HIDDEN_DIM),
                        layer_ids_to_inject=getattr(model.config, "layer_ids_to_inject", DEFAULT_FUSION_LAYER_IDS_TO_INJECT)
                    )
                
                print(f"model Dtype: {model.dtype}")
                # model = model.to(torch.bfloat16)
                # Print parameters after adding fusion components
                print("AFTER ADDING FUSION COMPONENTS:")
                print_up_A_parameters(model)
            else:
                from llava.model.language_model.llava_llama import LlavaConfig
                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                print('Loading LLaVA from base model...')
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print(f'Loading additional {name} weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables_skip_layer.bin')):
                non_lora_trainables_skip_layer = torch.load(os.path.join(model_path, 'non_lora_trainables_skip_layer.bin'), map_location='cpu')
            else:
                # That means there is no skip layer
                non_lora_trainables_skip_layer = None
                # # this is probably from HF Hub
                # from huggingface_hub import hf_hub_download
                # non_lora_trainables_skip_layer = load_from_hf(model_path, 'non_lora_trainables_skip_layer.bin')

            # Process key names first
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            if non_lora_trainables_skip_layer is not None:
                non_lora_trainables_skip_layer = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables_skip_layer.items()}
                if any(k.startswith('model.model.') for k in non_lora_trainables_skip_layer):
                    non_lora_trainables_skip_layer = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables_skip_layer.items()}
            
            # Print each name in non_lora_trainables_skip_layer and non_lora_trainables, along with their dtype
            if non_lora_trainables_skip_layer is not None:
                for k, v in non_lora_trainables_skip_layer.items():
                    print(f"non_lora_trainables_skip_layer[{k}]: {v.dtype}")
                    if "alpha" in k.lower():
                        print(f"alpha: {v}")
            for k, v in non_lora_trainables.items():
                print(f"non_lora_trainables[{k}]: {v.dtype}")
                if "alpha" in k.lower():
                    print(f"alpha: {v}")

            # Now load the state dict
            model.load_state_dict(non_lora_trainables, strict=False)
            if non_lora_trainables_skip_layer is not None:
                model.load_state_dict(non_lora_trainables_skip_layer, strict=False)
                # Print parameters after loading non_lora_trainables
                print("AFTER LOADING NON_LORA_TRAINABLES:")
                print_up_A_parameters(model)
                
                print("="*100)
                print(f"Keys in non_lora_trainables: {non_lora_trainables.keys()}")
                print("="*100)

            # Check the dtype of the non_lora_trainables
            print("="*100)
            # Try to print the dtype of the non_lora_trainables
            if 'model.layers.0.self_attn.q_proj.up_A' in non_lora_trainables:
                print(f"Dtype of non_lora_trainables: {non_lora_trainables['model.layers.0.self_attn.q_proj.up_A'].dtype}")
            print("="*100)

            # Check after loading non_lora_trainables, the values of each parameter loaded from non_lora_trainables are the same as the values in non_lora_trainables
            print("Verifying non_lora_trainables were loaded correctly...")
            for key, tensor in non_lora_trainables.items():
                if key.startswith('model.'):
                    param_name = key
                else:
                    param_name = 'model.' + key
                
                # Get the parameter from the model
                model_param = None
                for name, param in model.named_parameters():
                    if name == param_name:
                        model_param = param
                        break
                
                if model_param is not None:
                    # Check if the tensor values match
                    if not torch.allclose(model_param.cpu(), tensor.to(model_param.dtype).cpu(), rtol=1e-3, atol=1e-3):
                        print(f"Warning: Parameter {key} values differ between non_lora_trainables and loaded model")
                        print(f"  - non_lora_trainables: {tensor.mean().item():.6f} (mean), {tensor.std().item():.6f} (std), dtype: {tensor.dtype}")
                        print(f"  - model parameter: {model_param.mean().item():.6f} (mean), {model_param.std().item():.6f} (std), dtype: {model_param.dtype}")
                        
                        # Check if this might be a bfloat16 to float16 conversion issue
                        if tensor.dtype == torch.bfloat16 and model_param.dtype == torch.float16:
                            print(f"  - Note: This may be due to bfloat16 -> float16 conversion. Max abs diff: {(model_param.cpu() - tensor.to(model_param.dtype).cpu()).abs().max().item():.6f}")
                            
                            # Only check for extreme outliers that could cause serious issues
                            if (model_param.cpu() - tensor.to(model_param.dtype).cpu()).abs().max().item() > 0.1:
                                print(f"  - WARNING: Large discrepancy detected that may affect model performance")
                    else:
                        print(f"Parameter {key} loaded correctly")
                else:
                    print(f"Warning: Parameter {key} from non_lora_trainables not found in model")
            print("Verification complete")

            # Simply move layer 0 to match the device of layer 1
            if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 1 and non_lora_trainables_skip_layer is not None:
                # Get the target device from layer 1
                layer1_device = next(model.model.layers[1].parameters()).device
                
                # Move layer 0 to layer 1's device
                layer_ids_to_inject = [int(layer_id) for layer_id in model.config.layer_ids_to_inject.split(',')] 
                print(f"Moving layer {model.config.layer_ids_to_inject} to device: {layer1_device}")
                # Get the layers to inject fusion
                for layer_id in layer_ids_to_inject:
                    model.model.layers[layer_id] = model.model.layers[layer_id].to(layer1_device)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            name = "Fusion" if 'fusion' in model_name.lower() else "LLaVA"
            print(f'Loading {name} from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                if name == "Fusion":
                    # Force the config of load_from_lora to be false
                    cfg_pretrained.load_from_lora = False
                    model = FusionLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            
            if not "no-mm-proj" in model_name.lower():
                mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
                mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
                model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if 'fusion' in model_name.lower():
                    model = FusionLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
                    # model = get_fusion_model(
                    #     model,
                    #     vision_dim=getattr(model.config, "mm_hidden_size", DEFAULT_FUSION_VISION_DIM),
                    #     fusion_targets=getattr(model.config, "fusion_targets", DEFAULT_FUSION_TARGETS),
                    #     alpha=getattr(model.config, "fusion_alpha", DEFAULT_FUSION_ALPHA),
                    #     dropout=getattr(model.config, "fusion_dropout", DEFAULT_FUSION_DROPOUT),
                    # )
                    # # Load the non_lora_trainables_skip_layer
                    # non_lora_trainables_skip_layer = torch.load(os.path.join(model_path, 'non_lora_trainables_skip_layer.bin'), map_location='cpu')
                    # model.load_state_dict(non_lora_trainables_skip_layer, strict=False)
                else:
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        **kwargs
                    )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
