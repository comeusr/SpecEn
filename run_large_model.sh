#!/bin/bash

DEVICE=0,1
GPUS=2
DATA=gsm8k
SIZE=tiny
TARGET_MODEL=Qwen3-14B
METHOD=large_model
TEMPERATURE=0.0

mkdir -p ./logs/$DATA/${TARGET_MODEL}_${DRAFT_MODEL}/
mkdir -p ./generations/$DATA/${TARGET_MODEL}/${METHOD}_${TEMPERATURE}

export MODEL_PATH=/home/sagemaker-user/efs/model

CUDA_VISIBLE_DEVICES=$DEVICE \
    python ./main_dataset.py \
    dataset.name=$DATA \
    dataset.size=$SIZE \
    method=$METHOD \
    save_path="generations/$DATA/${TARGET_MODEL}/${METHOD}_${TEMPERATURE}" \
    method.model=$TARGET_MODEL \
    method.generate.temperature=$TEMPERATURE \
    method.generate.max_tokens=320 \
    method.llm.tensor_parallel_size=$GPUS \
    method.llm.max_model_len=2048
