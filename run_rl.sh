#!/bin/bash
LR=5e-4
DATA=gsm8k
LOSS=reinforce
MODEL=qwen
EPOCH=1
LAMBDA=10
TARGET_DRAFT_W=0.3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

python rl_train.py loss=$LOSS model=$MODEL datasets=[$DATA] \
    exp_name=Qwen3-8B_Qwen-06B_${LOSS}_equal_l2_reg_scale${LAMBDA}_target${TARGET_DRAFT_W}_${LR} \
    lr=${LR} \
    global_epochs=$EPOCH \
    n_examples=100 \
    wandb.project=Ensemble \
    model.name_or_path="Qwen/Qwen3-8B" \
    model.max_prompt_length=1152 \
    model.max_length=1536 \
    model.max_tokens=352 \
    cache_dir=/home/sagemaker-user/data/model \
    model.use_peft=false model.batch_size=2 \
    model.gradient_accumulation_steps=4 \
    model.reg_scale=${LAMBDA} \
    model.target_w_draft=${TARGET_DRAFT_W} \
    model.save_freqs=4

python -m train.generate \
    --model_path /home/sagemaker-user/data/model/Qwen3-8B_Qwen-06B_${LOSS}_equal_l2_reg_scale${LAMBDA}_targetdraft${TARGET_DRAFT_W}_${LR}/FINAL \
    --dataset $DATA --split test --temperature 0.0 \
    --max_tokens 352 --batch_size 8 --n_examples 200 \
    > .logs/$DATA/Qwen3-8B_Qwen-06B_${LOSS}_equal_l2_reg_scale${LAMBDA}_target${TARGET_DRAFT_W}_${LR}_FINAL.log 2>&1


