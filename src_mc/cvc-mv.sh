#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
python post_train.py --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_test \
    --do_lower_case \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size=4   \
    --per_gpu_train_batch_size=12  \
    --gradient_accumulation_steps 1 \
    --fp16 \
    --task_name RACE \
    --pre_model_dir 2020-03-23-14-25-checkpoint-118044-star \
    --output_dir ../output_mc
