#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation \
#     --model-base lmsys/vicuna-7b-v1.5 \
#     --question-file ./playground_temp/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder ./playground_temp/data/eval/vizwiz/test \
#     --answers-file ./playground_temp/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground_temp/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground_temp/data/eval/vizwiz/answers/llava-v1.5-7b.jsonl \
    --result-upload-file ./playground_temp/data/eval/vizwiz/answers_upload/llava-v1.5-7b.json
