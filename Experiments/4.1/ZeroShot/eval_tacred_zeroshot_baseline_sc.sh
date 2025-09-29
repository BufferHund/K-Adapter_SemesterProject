#!/bin/bash
# EVALUATION (TACRED): Zero-shot performance of roberta-large (baseline).
echo "====== EVALUATING (TACRED): ZERO-SHOT BASELINE ======"
python examples/run_finetune_TACRED_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=tacred \
    --do_eval \
    --data_dir=data/tacred  \
    --output_dir=./proc_data  \
    --comment 'eval_tacred_zeroshot_baseline' \
    --max_seq_length=184  \
    --per_gpu_eval_batch_size=8   \
    --overwrite_output_dir   \
    --overwrite_cache \
    --fusion_mode 'add'
