#!/usr/bin/env python

import os
import sys
import torch
import argparse
import json
import deepspeed
from transformers import AutoTokenizer
from PIL import Image
import jsonlines
from pathlib import Path

from llava.model import FusionLlavaForCausalLM
from llava.mm_utils import process_images
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

def parse_args():
    parser = argparse.ArgumentParser(description="Test fusion model with DeepSpeed and LoRA")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model or base model")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing images")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to test with")
    
    # Training arguments
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true", help="Use float16 precision")
    parser.add_argument("--deepspeed", type=str, help="DeepSpeed config file")
    parser.add_argument("--output-dir", type=str, default="./outputs/fusion_test")
    
    # LoRA arguments
    parser.add_argument("--lora-enable", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora-bias", type=str, default="none", help="LoRA bias")
    parser.add_argument("--lora-target-modules", type=str, default='["q_proj","k_proj","v_proj","o_proj"]', 
                      help="LoRA target modules")
    parser.add_argument("--lora-layer-ids-to-skip", type=str, default='0', 
                      help="LoRA layer IDs to skip (comma-separated)")
    
    # Vision Tower arguments
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--mm-vision-select-layer", type=int, default=-2)
    parser.add_argument("--pretrain-mm-mlp-adapter", type=str, default=None, 
                      help="Path to pretrained mm projector")
    
    # Fusion arguments
    parser.add_argument("--use-vision-fusion", action="store_true", default=True)
    parser.add_argument("--fusion-alpha", type=float, default=8.0)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument("--fusion-targets", type=str, default='["q","k","v"]')
    
    return parser.parse_args()

def load_data(args):
    """Load a small subset of data for testing"""
    print(f"Loading data from {args.data_path}")
    
    data = []
    if args.data_path.endswith('.json'):
        with open(args.data_path, 'r') as f:
            all_data = json.load(f)
            data = all_data[:args.num_samples]
    elif args.data_path.endswith('.jsonl'):
        with jsonlines.open(args.data_path) as reader:
            for i, item in enumerate(reader):
                if i >= args.num_samples:
                    break
                data.append(item)
    
    return data

def preprocess_data(data, tokenizer, image_folder, max_length=2048):
    """Preprocess the data for training"""
    processed_data = []
    
    for item in data:
        # Check if this is a conversation format or simple prompt-response format
        if "conversations" in item:
            # Assuming LLaVA conversation format
            conversation = item["conversations"]
            
            # Extract prompt and response
            prompt = ""
            response = ""
            for turn in conversation:
                if turn["from"] == "human":
                    prompt = turn["value"]
                elif turn["from"] == "gpt":
                    response = turn["value"]
        elif "prompt" in item:
            # Simple prompt-response format
            prompt = item["prompt"]
            response = item["response"] if "response" in item else ""
        
        # Process image
        image_file = None
        if "image" in item:
            image_file = item["image"]
        
        # Ensure image token is in prompt
        if DEFAULT_IMAGE_TOKEN not in prompt and image_file is not None:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        
        # Tokenize
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        
        # Create input_ids with both prompt and response
        input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
        
        # Create labels (-100 for prompt tokens so they're ignored in loss)
        labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
        
        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
        
        processed_item = {
            "input_ids": input_ids,
            "labels": labels,
            "image_file": image_file
        }
        processed_data.append(processed_item)
    
    return processed_data

def collate_fn(batch, tokenizer, image_processor, model_config, image_folder):
    """Collate function to prepare batch for training"""
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    
    # Pad input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    
    # Process images
    images = []
    for item in batch:
        if item["image_file"] is not None:
            image_path = os.path.join(image_folder, item["image_file"])
            image = Image.open(image_path).convert('RGB')
            images.append(image)
        else:
            images.append(None)
    
    # Only process images if at least one exists
    if any(img is not None for img in images):
        valid_images = [img for img in images if img is not None]
        if valid_images:
            image_tensors = process_images(valid_images, image_processor, model_config)
            
            # Assign image tensor to each item that had an image
            image_idx = 0
            for i, img in enumerate(images):
                if img is not None:
                    images[i] = image_tensors[image_idx]
                    image_idx += 1
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "images": images
    }

def check_fusion_layer_grads(model):
    """Check if gradients are flowing to fusion layers"""
    fusion_params = []
    
    # Collect fusion layer parameters
    if hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        first_layer = model.model.layers[0].self_attn
        
        for name in ['q_proj', 'k_proj', 'v_proj']:
            if hasattr(first_layer, name) and hasattr(getattr(first_layer, name), 'lora_A'):
                fusion_params.append((f"{name}.lora_A", getattr(first_layer, name).lora_A))
                fusion_params.append((f"{name}.lora_B", getattr(first_layer, name).lora_B))
    
    grad_info = []
    for name, param in fusion_params:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            grad_info.append(f"  {name}: No gradient!")
    
    return grad_info

def check_lora_layer_grads(model):
    """Check if gradients are flowing to LoRA layers"""
    from peft.tuners.lora import LoraLayer
    
    lora_params = []
    
    # Skip first layer as we're using fusion there
    for i, layer in enumerate(model.model.layers):
        if i == 0:  # Skip the first layer which uses fusion
            continue
            
        for name, module in layer.named_modules():
            if isinstance(module, LoraLayer):
                for param_name, param in module.named_parameters():
                    if "lora_" in param_name:
                        full_name = f"layer_{i}.{name}.{param_name}"
                        lora_params.append((full_name, param))
    
    grad_info = []
    
    # Sample a few parameters to check (to avoid very long outputs)
    sampled_params = lora_params[:5]  # Just check first 5
    
    for name, param in sampled_params:
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info.append(f"  {name}: grad_norm = {grad_norm:.6f}")
        else:
            grad_info.append(f"  {name}: No gradient!")
    
    return grad_info

def main():
    args = parse_args()
    
    # Setup DeepSpeed
    deepspeed.init_distributed()
    
    # Set local_rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    args.local_rank = local_rank
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    
    # Determine dtype
    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    
    # Fusion model config
    model_config = {
        "use_vision_fusion": args.use_vision_fusion,
        "fusion_alpha": args.fusion_alpha,
        "fusion_dropout": args.fusion_dropout,
        "fusion_targets": json.loads(args.fusion_targets),
        "mm_vision_tower": args.vision_tower,
        "mm_vision_select_layer": args.mm_vision_select_layer,
    }
    
    # Load model with fusion settings
    model = FusionLlavaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        **model_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    
    # Initialize vision components
    if args.pretrain_mm_mlp_adapter:
        mm_projector_weights = torch.load(args.pretrain_mm_mlp_adapter, map_location="cpu")
        model.get_model().mm_projector.load_state_dict(mm_projector_weights)
    
    image_processor = model.get_vision_tower().image_processor
    
    # Apply LoRA
    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        # Configure which layers to skip (typically the first layer with fusion)
        lora_layer_ids_to_skip = [int(x) for x in args.lora_layer_ids_to_skip.split(',')]
        target_modules = json.loads(args.lora_target_modules)
        
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Target modules: {target_modules}")
        print(f"Skipping layers: {lora_layer_ids_to_skip}")
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=["mm_projector"],
        )
        
        model = get_peft_model(model, lora_config)
        
    # Load a small subset of data
    data = load_data(args)
    print(f"Loaded {len(data)} samples for testing")
    
    # Preprocess data
    processed_data = preprocess_data(data, tokenizer, args.image_folder)
    
    # Create dataloader
    from torch.utils.data import DataLoader, Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Custom collate function
    def collate_wrapper(batch):
        return collate_fn(batch, tokenizer, image_processor, model.config, args.image_folder)
    
    # Create dataset and dataloader
    dataset = SimpleDataset(processed_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_train_batch_size,
        collate_fn=collate_wrapper
    )
    
    # Set up DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad]
    )
    
    # Set model to training mode
    model_engine.train()
    
    print("\n=== Testing fusion model with DeepSpeed and LoRA ===\n")
    
    # Run a few optimization steps
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Just test with 3 batches
            break
            
        print(f"\nProcessing batch {batch_idx+1}")
        
        # Move tensors to device
        input_ids = batch["input_ids"].to(model_engine.device)
        attention_mask = batch["attention_mask"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        
        # Convert list of image tensors to a tensor if available
        images = None
        if batch["images"] and any(img is not None for img in batch["images"]):
            valid_images = [img for img in batch["images"] if img is not None]
            if valid_images:
                images = torch.stack(valid_images).to(model_engine.device, dtype=dtype)
        
        print(f"  Input shape: {input_ids.shape}")
        if images is not None:
            print(f"  Images shape: {images.shape}")
        
        # Forward pass
        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images
        )
        
        loss = outputs.loss
        print(f"  Loss: {loss.item():.4f}")
        
        # Backward pass with DeepSpeed
        model_engine.backward(loss)
        
        # Check fusion layer gradients
        if batch_idx == 0:
            print("\n  Fusion layer gradients:")
            grad_info = check_fusion_layer_grads(model)
            for info in grad_info:
                print(info)
            
            if args.lora_enable:
                print("\n  LoRA layer gradients (sample):")
                lora_grad_info = check_lora_layer_grads(model)
                for info in lora_grad_info:
                    print(info)
        
        # Optimizer step with DeepSpeed
        model_engine.step()
        
        print(f"  Completed forward, backward, and optimization step successfully.")
    
    # Save a checkpoint
    checkpoint_path = os.path.join(args.output_dir, "test_checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    if local_rank == 0:
        print(f"\nSaving test checkpoint to {checkpoint_path}")
        model_engine.save_checkpoint(checkpoint_path)
    
    print("\n=== Test completed successfully! ===")
    print("Your fusion model is properly set up with DeepSpeed and LoRA.")
    print("You can now proceed with full training using the finetune_lora.sh script.")

if __name__ == "__main__":
    main() 