#!/bin/bash
# MODEL_PATH=$1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

python -m train.generate --model_path /home/sagemaker-user/data/model/REINFORCE/Qwen3-8B_Qwen-06B_reinforce_5e-4/47 \
    --dataset gsm8k --split test --temperature 0.0 \
    --max_tokens 1024 --batch_size 4 --n_examples 200