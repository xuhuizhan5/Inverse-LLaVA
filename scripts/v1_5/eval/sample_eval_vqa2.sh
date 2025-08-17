#!/bin/bash

# GPU configuration
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# Model and dataset configuration
CKPT="llava-v1.5-7b"
SPLIT="llava_vqav2_mscoco_test-dev2015"
SAMPLE_SIZE=40  # Number of samples to evaluate
OUTPUT_DIR="./playground_temp/data/eval/vqav2/samples/${SPLIT}/${CKPT}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Use the first GPU in the list
CUDA_VISIBLE_DEVICES=${GPULIST[0]} python -m llava.eval.model_vqa_sample_eval \
    --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground_temp/data/eval/vqav2/$SPLIT.jsonl \
    --image-folder ./playground_temp/data/eval/vqav2/test2015 \
    --answers-file "$OUTPUT_DIR/answers.jsonl" \
    --sample-size $SAMPLE_SIZE \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    # --random_sample

echo "Sample evaluation completed. Results saved to $OUTPUT_DIR"

