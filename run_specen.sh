#!/bin/bash

DATA=gsm8k
METHOD=sd
CLASS=meta-llama
DRAFT=Llama-3.2-1B-Instruct
TARGET=Llama-3.1-8B-Instruct

export TRANSFORMERS_VERBOSITY=error

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

python speculative_decoding.py \
    --method ${METHOD} \
    --draft_model ${CLASS}/${DRAFT} \
    --target_model ${CLASS}/${TARGET} \
    --model_path /home/sagemaker-user/data/model/Llama-3.1-8B-Instruct_Llama-3.2-1B-Instruct_reinforce_equal_l2_reg_scale10_target0.3_5e-4/48 \
    --dataset $DATA --split test --temperature 0.6 \
    --max_tokens 354 --batch_size 1 --n_examples 200