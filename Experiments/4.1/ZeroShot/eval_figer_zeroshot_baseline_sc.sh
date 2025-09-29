#!/bin/bash
# EVALUATION (FIGER): Zero-shot performance of roberta-large (baseline).
echo "====== EVALUATING (FIGER): ZERO-SHOT BASELINE ======"
python examples/run_finetune_figer_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=entity_type \
    --do_eval \
    --data_dir=data/FIGER  \
    --output_dir=./proc_data  \
    --comment 'eval_figer_zeroshot_baseline' \
    --max_seq_length=256  \
    --per_gpu_eval_batch_size=8   \
    --overwrite_output_dir   \
    --overwrite_cache \
    --fusion_mode 'add'
