#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python main.py --model_type bert \
    --model_name_or_path bert-base-uncased \
    --do_lower_case \
    --max_seq_length 384 \
    --per_gpu_eval_batch_size 8  \
    --do_test \
    --eval_all_checkpoints \
    --task_name RACE \
    --time_stamp 03-23-14-25

