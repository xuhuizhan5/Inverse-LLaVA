#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground_temp/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground_temp/data/eval/scienceqa/images/test \
    --answers-file ./playground_temp/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground_temp/data/eval/scienceqa \
    --result-file ./playground_temp/data/eval/scienceqa/answers/llava-v1.5-7b.jsonl \
    --output-file ./playground_temp/data/eval/scienceqa/answers/llava-v1.5-7b_output.jsonl \
    --output-result ./playground_temp/data/eval/scienceqa/answers/llava-v1.5-7b_result.json
