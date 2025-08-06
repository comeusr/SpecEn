#!/bin/bash

DATA=wmt
METHOD=static
STATIC_DRAFT=0.0
CLASS=meta-llama
DRAFT=Llama-3.2-1B-Instruct
TARGET=Llama-3.1-8B-Instruct
ITER=10
DO_SAMPLE=False

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

python -m train.generate \
    --model_path /home/ubuntu/kasasiva_exps_DONOT_DELETE/ziyi_codes_17July2025/data/${DATA}/model/Llama-3.1-8B-Instruct \
    --dataset $DATA --split validation \
    --do_sample ${DO_SAMPLE} --temperature 0.0 \
    --method ${METHOD} --static_draft_weights ${STATIC_DRAFT} \
    --draft_model ${CLASS}/${DRAFT} \
    --target_model ${CLASS}/${TARGET} \
    --max_tokens 128 --batch_size 8 --n_examples 500 \
    # > .logs/$DATA/Llama-3.1-8B-Instruct.log 2>&1
