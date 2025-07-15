#!/bin/bash
# MODEL_PATH=$1
DATA=gsm8k

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

python speculative_decoding.py \
    --model_path /home/sagemaker-user/data/model/Qwen3-8B_Qwen-06B_reinforce_equal_init_scale20_5e-4/48 \
    --dataset $DATA --split test --temperature 0.1 \
    --max_tokens 1024 --batch_size 1 --n_examples 200 \
