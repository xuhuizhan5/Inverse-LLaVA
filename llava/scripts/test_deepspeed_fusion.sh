#!/bin/bash

# Test DeepSpeed fusion training script

WORLD_SIZE=1  # Change for multi-GPU
MODEL_PATH="./checkpoints/llama-2-7b-chat"  # Adjust path
DATA_PATH="./playground/data/llava_instruct_80k.json"  # Adjust path
IMAGE_FOLDER="/path/to/coco/train2017"  # Adjust path
NUM_SAMPLES=10
OUTPUT_DIR="./outputs/fusion_test"

ZERO_STAGE=2  # DeepSpeed ZeRO stage
DEEPSPEED_CONFIG="./scripts/zero${ZERO_STAGE}.json"  # DeepSpeed config

# Create minimal dataset for testing
python scripts/create_small_dataset.py \
  --input-file $DATA_PATH \
  --output-file "$DATA_PATH.small" \
  --num-samples $NUM_SAMPLES

# Run DeepSpeed fusion test
deepspeed --num_gpus=$WORLD_SIZE scripts/test_deepspeed_fusion.py \
  --deepspeed $DEEPSPEED_CONFIG \
  --model-path $MODEL_PATH \
  --data-path "$DATA_PATH.small" \
  --image-folder $IMAGE_FOLDER \
  --num-samples $NUM_SAMPLES \
  --output-dir $OUTPUT_DIR \
  --lora-enable \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-target-modules '["q_proj","k_proj","v_proj","o_proj"]' \
  --lora-layer-ids-to-skip '0' \
  --use-vision-fusion \
  --fusion-alpha 8.0 \
  --fusion-dropout 0.1 \
  --fusion-targets '["q","k","v"]' \
  --vision-tower "openai/clip-vit-large-patch14" \
  --mm-vision-select-layer -2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --bf16 