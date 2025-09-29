#!/bin/bash
# EXPERIMENT: Baseline (full fine-tuning)
echo "====== RUNNING EXPERIMENT: BASELINE ======"
python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=entity_type \
    --do_train --do_eval \
    --data_dir=data/OpenEntity \
    --output_dir=./proc_data  \
    --comment 'exp_baseline' \
    --max_seq_length=256  \
    --per_gpu_train_batch_size=8   \
    --learning_rate=2e-5 \
    --max_steps=12000  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=120 \
    --save_steps=2000 \
    --freeze_bert="" \
    --fusion_mode 'add'
