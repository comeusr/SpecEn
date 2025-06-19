#!/bin/bash

DEVICE=$1
GPUS=$2
DATA=$3
SIZE=small
TARGET_MODEL=$4
DRAFT_MODEL=$5
METHOD=sd
GAMMA=$6
ENSEMBLE=$7
ALPHA=$8
TEMPERATURE=$9
MAX_TOKENS=${10}

mkdir -p ./logs/$DATA/${TARGET_MODEL}_${DRAFT_MODEL}/
mkdir -p ./generations/$DATA/${TARGET_MODEL}_${DRAFT_MODEL}/${METHOD}_${ENSEMBLE}_${MAX_TOKENS}_${GAMMA}_${ALPHA}_${TEMPERATURE}

export MODEL_PATH=/home/sagemaker-user/efs/model

CUDA_VISIBLE_DEVICES=$DEVICE \
    python ./main_dataset.py \
    dataset.name=$DATA \
    dataset.size=$SIZE \
    save_path="generations/$DATA/${TARGET_MODEL}_${DRAFT_MODEL}/${METHOD}_${ENSEMBLE}_${MAX_TOKENS}_${GAMMA}_${ALPHA}_${TEMPERATURE}" \
    method=$METHOD \
    method.gamma=$GAMMA \
    method.model=$TARGET_MODEL \
    method.draft_model=$DRAFT_MODEL \
    method.ensemble=$ENSEMBLE \
    method.alpha=$ALPHA \
    method.generate.temperature=$TEMPERATURE \
    method.generate.max_tokens=$MAX_TOKENS \
    method.llm.tensor_parallel_size=$GPUS \
    method.llm.max_model_len=4096 > ./logs/$DATA/${TARGET_MODEL}_${DRAFT_MODEL}/${METHOD}_${ENSEMBLE}_${MAX_TOKENS}_${GAMMA}_${ALPHA}_${TEMPERATURE}.log
