#!/bin/bash
LR=5e-4
DATA=gsm8k
LOSS=sft
MODEL=qwen
EPOCH=2

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

accelerate launch \
    --config_file accelerate_config/ddp_4gpu.yaml \
    --main_process_port $MASTER_PORT \
    sft_train.py loss=$LOSS model=$MODEL datasets=[$DATA] exp_name=Qwen3-8B_Qwen-06B_${LOSS}_${LR} \
    global_epochs=$EPOCH \
    wandb.project=Ensemble \
    model.name_or_path="Qwen/Qwen3-8B" \
    model.max_prompt_length=1024 \
    cache_dir=/home/sagemaker-user/data/model \
    model.use_peft=false lr=${LR} model.batch_size=4
    
    