#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground_temp/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground_temp/data/eval/textvqa/train_images \
    --answers-file ./playground_temp/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground_temp/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground_temp/data/eval/textvqa/answers/llava-v1.5-7b.jsonl
