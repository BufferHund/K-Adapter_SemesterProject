#!/bin/bash
# EXPERIMENT: fac+lin adapters with CONCAT fusion (ORIGINAL HPs)
# This is the configuration that produced poor results.
python examples/run_finetune_openentity_adapter.py \
    --model_type roberta \
    --model_name_or_path roberta-large \
    --task_name=entity_type \
    --do_train --do_eval \
    --data_dir=data/OpenEntity \
    --output_dir=./proc_data \
    --comment 'exp_fac_lin_concat_original_hp' \
    --max_seq_length=256 \
    --per_gpu_train_batch_size=4 \
    --learning_rate=1e-4 \
    --max_steps=12000 \
    --overwrite_output_dir \
    --overwrite_cache \
    --warmup_steps=120 \
    --save_steps=2000 \
    --freeze_bert="" \
    --freeze_adapter="True" \
    --adapter_size 768 \
    --adapter_list "0,11,22" \
    --adapter_skip_layers 0 \
    --meta_fac_adaptermodel="./pretrained_models/fac-adapter/pytorch_model.bin" \
    --meta_lin_adaptermodel="./pretrained_models/lin-adapter/pytorch_model.bin" \
    --fusion_mode 'concat'
