#!/bin/bash

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

python -m train.generate --model_path /home/sagemaker-user/data/model/Qwen3-8B_Qwen-06B_sft_5e-4/FINAL \
    --dataset gsm8k --split test --temperature 0.0 \
    --max_tokens 1024 --batch_size 4