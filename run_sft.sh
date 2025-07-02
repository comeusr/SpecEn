#!/bin/bash

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

accelerate launch \
    --config_file accelerate_config/fsdp_4gpu.yaml \
    --main_process_port $MASTER_PORT \
    sft_train.py loss=sft model=qwen datasets=[gsm8k] exp_name=Qwen3-8B_Qwen-06B_sft \
    wandb.project=Ensemble \
    model.name_or_path="Qwen/Qwen3-8B" \
    model.use_peft=false lr=1e-6 model.batch_size=4
    
    