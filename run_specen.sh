#!/bin/bash

DATA=gsm8k
METHOD=auto
CLASS=meta-llama
DRAFT=Llama-3.2-1B-Instruct
TARGET=Llama-3.2-3B-Instruct
STATIC_EW=0.15
ITER=FINAL
NUM_DRAFT=10
BATCH_SIZE=16

# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found. Please create it with your API tokens."
  exit 1
fi
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

mkdir -p .logs/$DATA

python speculative_decoding.py \
    --method ${METHOD} \
    --draft_model ${CLASS}/${DRAFT} \
    --target_model ${CLASS}/${TARGET} \
    --model_path ../data/model/Llama-3.1-8B-Instruct_Llama-3.2-1B-Instruct_reinforce_equal_l2_reg_scale5_target0.15_5e-4/${ITER} \
    --dataset $DATA --split test --temperature 0.0 \
    --draft_ensemble_weights ${STATIC_EW} \
    --num_assistant_tokens ${NUM_DRAFT} \
    --max_tokens 354 --batch_size ${BATCH_SIZE} --n_examples 200 \
    > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}.log 2>&1







# #!/bin/bash

# DATA=gsm8k
# METHOD=sd
# CLASS=meta-llama
# DRAFT=Llama-3.2-1B-Instruct
# TARGET=Llama-3.1-8B-Instruct

# # Load environment variables from .env file
# if [ -f .env ]; then
#   export $(grep -v '^#' .env | xargs)
# else
#   echo "Error: .env file not found. Please create it with your API tokens."
#   exit 1
# fi
# export TRANSFORMERS_VERBOSITY=error

# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=$(shuf -i 29500-29600 -n 1)

# wandb login $WANDB_API_KEY
# huggingface-cli login --token $HF_TOKEN

# mkdir -p .logs/$DATA

# python speculative_decoding.py \
#     --method ${METHOD} \
#     --draft_model ${CLASS}/${DRAFT} \
#     --target_model ${CLASS}/${TARGET} \
#     --model_path /home/sagemaker-user/data/model/Llama-3.1-8B-Instruct_Llama-3.2-1B-Instruct_reinforce_equal_l2_reg_scale10_target0.3_5e-4/48 \
#     --dataset $DATA --split test --temperature 0.6 \
#     --max_tokens 354 --batch_size 1 --n_examples 200
