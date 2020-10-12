#!/usr/bin/env bash
# For RACE
export CUDA_VISIBLE_DEVICES=0,1
python main.py --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 36 \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=6   \
    --gradient_accumulation_steps 2 \
    --fp16 \
    --task_name RACE \
    --seed 122