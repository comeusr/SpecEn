#!/bin/bash

DATA=xsum
METHOD=auto
# CLASS=meta-llama
CLASS=google
DRAFT=Llama-3.2-1B-Instruct
# TARGET=Llama-3.1-8B-Instruct
TARGET=gemma-3-12b-it
STATIC_EW=0.0
TEMPERATURE=0.1 # kasasiva - changed from 0.7 to 1.0
ITER=264
NUM_DRAFT=5
N_EXAMPELS=5 #kasasiva - change to 5 for quick test, otherwise keep it 200
DO_SAMPLE=True

export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
export HF_TOKEN=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
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

#kasasiva - do change the model_path argument as per below
# if RL trained Ensemblehead - copy the name from the folder where the model is saved
# otherwise the change the name according to the technique as there is no concept of saved model here. Remove the {ITER} in this case.
# similarly remove the {ITER} in the logs and rename accordingly.
# double check if your particular dataset has 'test' split, otherwise 'validation' split

python speculative_decoding.py \
    --method ${METHOD} --do_sample ${DO_SAMPLE}\
    --draft_model ${CLASS}/${DRAFT} \
    --target_model ${CLASS}/${TARGET} \
    --model_path ../data/${DATA}/model/gemma-12b-Instruct_autoregressivedecoding/ \
    --dataset $DATA --split test --temperature ${TEMPERATURE} \
    --draft_ensemble_weights ${STATIC_EW} \
    --num_assistant_tokens ${NUM_DRAFT} \
    --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPELS} \
    > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}.log 2>&1

# python speculative_decoding.py \
#     --method ${METHOD} --do_sample ${DO_SAMPLE}\
#     --draft_model ${CLASS}/${DRAFT} \
#     --target_model ${CLASS}/${TARGET} \
#     --model_path ../data/${DATA}/model/Llama-3.1-8B-Instruct_autoregressivedecoding/ \
#     --dataset $DATA --split test --temperature ${TEMPERATURE} \
#     --draft_ensemble_weights ${STATIC_EW} \
#     --num_assistant_tokens ${NUM_DRAFT} \
#     --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPELS} \
#     > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}.log 2>&1


# DATA=cnndm
# METHOD=sd
# CLASS=meta-llama
# DRAFT=Llama-3.2-1B-Instruct
# TARGET=Llama-3.1-8B-Instruct
# STATIC_EW=0.0
# TEMPERATURE=0.7
# ITER=264
# NUM_DRAFT=5
# N_EXAMPELS=200
# DO_SAMPLE=True

# export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
# export HF_TOKEN=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
# export TRANSFORMERS_VERBOSITY=error

# wandb login $WANDB_API_KEY
# huggingface-cli login --token $HF_TOKEN

# mkdir -p .logs/$DATA

# python speculative_decoding.py \
#     --method ${METHOD} --do_sample ${DO_SAMPLE}\
#     --draft_model ${CLASS}/${DRAFT} \
#     --target_model ${CLASS}/${TARGET} \
#     --model_path ../data/${DATA}/model/Llama-3.1-8B-Instruct_Llama-3.2-1B-Instruct_TEMP0.7_reinforce_AdamW_reg_scale10_target0.3_5e-4/${ITER} \
#     --dataset $DATA --split test --temperature ${TEMPERATURE} \
#     --draft_ensemble_weights ${STATIC_EW} \
#     --num_assistant_tokens ${NUM_DRAFT} \
#     --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPELS} \
#     > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}.log 2>&1
