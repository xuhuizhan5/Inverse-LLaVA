#!/usr/bin/env python

import json
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Create a small dataset for testing")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the original dataset JSON file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the smaller dataset")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to include")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load the original dataset
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # Select a random subset
    if len(data) > args.num_samples:
        data_subset = random.sample(data, args.num_samples)
    else:
        data_subset = data
    
    # Save the smaller dataset
    with open(args.output_file, 'w') as f:
        json.dump(data_subset, f, indent=2)
    
    print(f"Created smaller dataset with {len(data_subset)} samples at {args.output_file}")

if __name__ == "__main__":
    main() 