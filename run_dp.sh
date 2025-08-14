#!/bin/bash

# Script to run speculative decoding with data parallelism
# Usage: ./run_dp.sh [num_gpus]

# Default to using all available GPUs if not specified

NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# Check if we have GPUs available
if [ $NUM_GPUS -lt 1 ]; then
    echo "Error: No GPUs detected. This script requires at least one GPU."
    exit 1
fi

echo "Running with $NUM_GPUS GPUs"

# Set common parameters
TARGET_MODEL='google/gemma-2-9b-it' # 'google/gemma-2-9b-it' # "meta-llama/Llama-3.2-3B-Instruct"
DRAFT_MODEL='google/gemma-2-2b-it' # 'google/gemma-2-2b-it' # "meta-llama/Llama-3.2-1B-Instruct"
METHOD="sd"  # Options: sd, sd_en, static_en
DATASET="cnndm"  # Options: gsm8k, cnndm, xsum
SPLIT="test"
N_EXAMPLES=200  # Number of examples to process
BATCH_SIZE=8    # Batch size per GPU
MAX_TOKENS=256  # Maximum tokens to generate
TEMPERATURE=0.2
DO_SAMPLE="False"
NUM_ASSISTANT_TOKENS=15
ASSISTANT_SCHEDULE='constant' # Options: dynamic, constant, heuristic
ASSISTANT_CONFIDENT_THRESHOLD=0
SEED=42
MODEL_PATH="data/${DATASET}/model/${TARGET_MODEL}_${DRAFT_MODEL}_speculativedecoding_DRAFT_LEN${NUM_ASSISTANT_TOKENS}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLE${DO_SAMPLE}_NEW"
DRAFT_ENSEMBLE_WEIGHTS=0.0  # Only used for static_en method

# Create the output directory if it doesn't exist
mkdir -p $MODEL_PATH

# Run the data parallel version using torchrun
torchrun --nproc_per_node=$NUM_GPUS \
    $(dirname "$0")/speculative_decoding_dp.py \
    --target_model $TARGET_MODEL \
    --draft_model $DRAFT_MODEL \
    --method $METHOD \
    --dataset $DATASET \
    --split $SPLIT \
    --n_examples $N_EXAMPLES \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --do_sample $DO_SAMPLE \
    --draft_len $NUM_ASSISTANT_TOKENS \
    --num_assistant_tokens $NUM_ASSISTANT_TOKENS \
    --seed $SEED \
    --model_path $MODEL_PATH \
    --assistant_schedule $ASSISTANT_SCHEDULE \
    --assistant_confidence_threshold $ASSISTANT_CONFIDENT_THRESHOLD \
    --draft_ensemble_weights $DRAFT_ENSEMBLE_WEIGHTS

#!/bin/bash

# Script to run speculative decoding with data parallelism
# Usage: ./run_dp.sh [num_gpus]

# Default to using all available GPUs if not specified

# Change by wzyi

# NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# # Check if we have GPUs available
# if [ $NUM_GPUS -lt 1 ]; then
#     echo "Error: No GPUs detected. This script requires at least one GPU."
#     exit 1
# fi

# echo "Running with $NUM_GPUS GPUs"

# GPU_DEVICES=${CUDA_VISIBLE_DEVICES:-"4,5,6,7"}
# export CUDA_VISIBLE_DEVICES=$GPU_DEVICES

# # Calculate number of GPUs from CUDA_VISIBLE_DEVICES
# NUM_GPUS=$(echo $GPU_DEVICES | tr ',' '\n' | wc -l)

# echo "Using GPUs: $GPU_DEVICES"
# echo "Number of GPUs: $NUM_GPUS"

# # Set common parameters
# TARGET_MODEL="meta-llama/Llama-3.2-3B-Instruct"
# DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
# METHOD="sd"  # Options: sd, sd_en, static_en
# DATASET="gsm8k"  # Options: gsm8k, cnndm, xsum
# SPLIT="test"
# N_EXAMPLES=32  # Number of examples to process
# BATCH_SIZE=8    # Batch size per GPU
# MAX_TOKENS=256  # Maximum tokens to generate
# TEMPERATURE=0.2
# DO_SAMPLE="False"
# DRAFT_LEN=5
# NUM_ASSISTANT_TOKENS=10
# SEED=42
# MODEL_PATH="data/${DATASET}/model/${TARGET_MODEL}_${DRAFT_MODEL}_speculativedecoding_DRAFT_LEN${DRAFT_LEN}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLE${DO_SAMPLE}"
# DRAFT_ENSEMBLE_WEIGHTS=0.0  # Only used for static_en method

# # Create the output directory if it doesn't exist
# mkdir -p $MODEL_PATH

# # Run the data parallel version using torchrun
# torchrun --nproc_per_node=$NUM_GPUS \
#     $(dirname "$0")/speculative_decoding_dp.py \
#     --target_model $TARGET_MODEL \
#     --draft_model $DRAFT_MODEL \
#     --method $METHOD \
#     --dataset $DATASET \
#     --split $SPLIT \
#     --n_examples $N_EXAMPLES \
#     --batch_size $BATCH_SIZE \
#     --max_tokens $MAX_TOKENS \
#     --temperature $TEMPERATURE \
#     --do_sample $DO_SAMPLE \
#     --draft_len $DRAFT_LEN \
#     --num_assistant_tokens $NUM_ASSISTANT_TOKENS \
#     --seed $SEED \
#     --model_path $MODEL_PATH \
#     --draft_ensemble_weights $DRAFT_ENSEMBLE_WEIGHTS
