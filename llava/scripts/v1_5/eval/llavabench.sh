#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/llava-v1.5-7b-fusion-lora-no-activation-no-text-proj \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground_temp/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground_temp/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground_temp/data/eval/llava-bench-in-the-wild/answers/stable-fusion-no-text-proj.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground_temp/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground_temp/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground_temp/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground_temp/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground_temp/data/eval/llava-bench-in-the-wild/answers/stable-fusion-no-text-proj.jsonl \
    --output \
        playground_temp/data/eval/llava-bench-in-the-wild/reviews/stable-fusion-no-text-proj.jsonl

python llava/eval/summarize_gpt_review.py -f playground_temp/data/eval/llava-bench-in-the-wild/reviews/stable-fusion-no-text-proj.jsonl
