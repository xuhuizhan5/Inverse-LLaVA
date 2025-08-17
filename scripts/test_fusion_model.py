import torch
from llava.model import FusionLlavaForCausalLM
from transformers import AutoTokenizer
from PIL import Image
from llava.mm_utils import process_images
import argparse

def test_fusion_model(args):
    # Load model
    model = FusionLlavaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Process a sample image
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = process_images([image], model.get_vision_tower().image_processor, model.config)
    
    # Create a simple prompt
    prompt = f"<image>\nDescribe this image."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate
    outputs = model.generate(
        input_ids,
        images=image_tensor.to(model.device, dtype=torch.float16),
        max_new_tokens=100
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    args = parser.parse_args()
    
    test_fusion_model(args) 