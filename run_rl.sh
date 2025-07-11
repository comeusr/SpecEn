#!/bin/bash
LR=5e-4
DATA=gsm8k
LOSS=reinforce
MODEL=qwen
EPOCH=1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

python rl_train.py loss=$LOSS model=$MODEL datasets=[$DATA] exp_name=Qwen3-8B_Qwen-06B_${LOSS}_${LR} \
    lr=${LR} \
    global_epochs=$EPOCH \
    n_examples=100 \
    wandb.project=Ensemble \
    model.name_or_path="Qwen/Qwen3-8B" \
    model.max_prompt_length=800 \
    cache_dir=/home/sagemaker-user/data/model/REINFORCE \
    model.use_peft=false model.batch_size=2 \
    model.gradient_accumulation_steps=4 \
    model.save_freqs=8