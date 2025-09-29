#!/bin/bash
# EXPERIMENT: fac-adapter only (optimal HPs)
echo "====== RUNNING EXPERIMENT: FAC-ADAPTER (OPTIMAL) ======"
python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=entity_type \
    --do_train --do_eval \
    --data_dir=data/OpenEntity \
    --output_dir=./proc_data  \
    --comment 'exp_fac_only_optimal' \
    --max_seq_length=256  \
    --per_gpu_train_batch_size=4   \
    --learning_rate=5e-6 \
    --max_steps=12000  \
    --overwrite_output_dir   \
    --overwrite_cache \
    --warmup_steps=500 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
    --fusion_mode 'add'
