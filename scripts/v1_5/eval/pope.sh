#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground_temp/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground_temp/data/eval/pope/val2014 \
    --answers-file ./playground_temp/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground_temp/data/eval/pope/coco \
    --question-file ./playground_temp/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground_temp/data/eval/pope/answers/llava-v1.5-7b.jsonl
