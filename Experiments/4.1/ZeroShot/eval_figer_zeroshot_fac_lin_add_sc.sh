#!/bin/bash
# EVALUATION (FIGER): Zero-shot performance of roberta-large with fac+lin adapters (ADD fusion).
python examples/run_finetune_figer_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=entity_type \
    --do_eval \
    --data_dir=data/FIGER  \
    --output_dir=./proc_data  \
    --comment 'eval_figer_zeroshot_fac_lin_add' \
    --max_seq_length=256  \
    --per_gpu_eval_batch_size=8   \
    --overwrite_output_dir   \
    --overwrite_cache \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
    --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
    --fusion_mode 'add'
