#!/bin/bash
# MODEL_PATH=$1
DATA=gsm8k

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

python -m train.generate \
    --model_path /home/sagemaker-user/data/model/Qwen3-8B \
    --dataset $DATA --split test --temperature 0.0 \
    --max_tokens 384 --batch_size 8 --n_examples 200 \
    > .logs/$DATA/Qwen3-8B.log 2>&1
