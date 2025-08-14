#!/bin/bash

# Script to run speculative decoding with Gemma models in data parallel setting for WMT dataset
# Usage: ./run_dp_sd_kasasiva.sh [num_gpus]

# Default to using all available GPUs if not specified
NUM_GPUS=${1:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# Check if we have GPUs available
if [ $NUM_GPUS -lt 1 ]; then
    echo "Error: No GPUs detected. This script requires at least one GPU."
    exit 1
fi

echo "Running with $NUM_GPUS GPUs"

# Model combinations (target, draft) - Gemma family
declare -A MODEL_PAIRS=(
    # ["gemma-3-12b-it"]="gemma-3-1b-it"
    ["gemma-3-12b-it"]="gemma-3-4b-it"
    # ["gemma-3-4b-it"]="gemma-3-1b-it"

    # Added by Ziyi, for testing purpose
    # ['Llama-3.1-8B-Instruct']='Llama-3.2-1B-Instruct'
)

# Draft lengths to try
DRAFT_LENGTHS=(3 5 7 9 11 13 15)

# Fixed parameters (same as run_sd_kasasiva.sh)
METHOD=sd
CLASS=google
# Added by Ziyi, for testing purpose 
# CLASS=meta-llama
STATIC_EW=0.0
TEMPERATURE=0.2
ITER=264
N_EXAMPLES=500
DO_SAMPLE=True
DATASET="wmt"  # Focus on WMT dataset only
SPLIT="validation" # for wmt it has bee to validation
BATCH_SIZE=8    # Batch size per GPU
MAX_TOKENS=128  # Maximum tokens to generate
SEED=42
ASSISTANT_SCHEDULE='constant' # Options: dynamic, constant, heuristic
ASSISTANT_CONFIDENT_THRESHOLD=0

# Authentication setup (same as kasasiva)
export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
export HF_TOKEN=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
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Create log directory for WMT dataset
mkdir -p .logs/$DATASET

# Loop over each model combination and draft length
for TARGET in "${!MODEL_PAIRS[@]}"; do
    DRAFT=${MODEL_PAIRS[$TARGET]}
    
    for NUM_DRAFT in "${DRAFT_LENGTHS[@]}"; do
        echo "Processing dataset: $DATASET with target: $TARGET, draft: $DRAFT, draft length: $NUM_DRAFT using $NUM_GPUS GPUs"
        
        # Set model path (same structure as kasasiva)
        MODEL_PATH="../data/${DATASET}/model/${TARGET}_${DRAFT}_sd_13Aug_ksr_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}/"
        
        # Create the output directory if it doesn't exist
        mkdir -p $MODEL_PATH
        
        # Run the data parallel version using torchrun
        torchrun --nproc_per_node=$NUM_GPUS \
            $(dirname "$0")/speculative_decoding_dp.py \
            --target_model ${CLASS}/${TARGET} \
            --draft_model ${CLASS}/${DRAFT} \
            --method $METHOD \
            --dataset $DATASET \
            --split $SPLIT \
            --n_examples $N_EXAMPLES \
            --batch_size $BATCH_SIZE \
            --max_tokens $MAX_TOKENS \
            --temperature $TEMPERATURE \
            --do_sample $DO_SAMPLE \
            --num_assistant_tokens $NUM_DRAFT \
            --seed $SEED \
            --model_path $MODEL_PATH \
            --draft_ensemble_weights $STATIC_EW \
            --gpu_count $NUM_GPUS \
            --assistant_schedule $ASSISTANT_SCHEDULE \
            --assistant_confidence_threshold $ASSISTANT_CONFIDENT_THRESHOLD \
            > .logs/${DATASET}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_13Aug_DRAFT_LEN${NUM_DRAFT}_${ITER}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}.log 2>&1
        
        echo "Completed ${DATASET} with target: ${TARGET}, draft: ${DRAFT}, draft length: ${NUM_DRAFT}"
    done
done

echo "All experiments completed for WMT dataset with Gemma models in data parallel setting!"
