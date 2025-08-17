import argparse
import torch
from llava.model import FusionLlavaForCausalLM

def save_fusion_model(args):
    model = FusionLlavaForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Save with proper config
    model.save_pretrained(args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    
    save_fusion_model(args) 