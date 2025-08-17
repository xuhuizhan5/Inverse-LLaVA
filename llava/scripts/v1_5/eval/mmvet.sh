#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-v1.5-7b-lora \
    --question-file ./playground_temp/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground_temp/data/eval/mm-vet/images \
    --answers-file ./playground_temp/data/eval/mm-vet/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground_temp/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground_temp/data/eval/mm-vet/answers/llava-v1.5-7b-lora.jsonl \
    --dst ./playground_temp/data/eval/mm-vet/results/llava-v1.5-7b-lora.json

cd /home/zhanx5/Unified-fusion/playground_temp/data/eval/mm-vet/
python mm-vet_evaluator.py \
    --result_file results/llava-v1.5-7b-lora.json \
    --mmvet_path /home/zhanx5/Unified-fusion/playground_temp/data/eval/mm-vet/
    

