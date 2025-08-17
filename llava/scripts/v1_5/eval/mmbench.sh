#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --question-file ./playground_temp/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground_temp/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground_temp/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground_temp/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground_temp/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground_temp/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b-lora
