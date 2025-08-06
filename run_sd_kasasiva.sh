#!/bin/bash

# Datasets to loop over
DATASETS=("cnndm" "wmt" "xsum")

# Model combinations (target, draft)
declare -A MODEL_PAIRS=(
    ["gemma-3-12b-it"]="gemma-3-1b-it"
    ["gemma-3-12b-it"]="gemma-3-4b-it"
    ["gemma-3-4b-it"]="gemma-3-1b-it"
)

# Draft lengths to try
DRAFT_LENGTHS=(3 5 7 9 11 13 15)

# Fixed parameters
METHOD=sd
CLASS=google
STATIC_EW=0.0
TEMPERATURE=0.2
ITER=264
N_EXAMPLES=500
DO_SAMPLE=True

export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
export HF_TOKEN=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
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Loop over each dataset, model combination, and draft length
for DATA in "${DATASETS[@]}"; do
    mkdir -p .logs/$DATA
    
    for TARGET in "${!MODEL_PAIRS[@]}"; do
        DRAFT=${MODEL_PAIRS[$TARGET]}
        
        for NUM_DRAFT in "${DRAFT_LENGTHS[@]}"; do
            echo "Processing dataset: $DATA with target: $TARGET, draft: $DRAFT, draft length: $NUM_DRAFT"
            
            python speculative_decoding.py \
                --method ${METHOD} --do_sample ${DO_SAMPLE} \
                --draft_model ${CLASS}/${DRAFT} \
                --target_model ${CLASS}/${TARGET} \
                --model_path ../data/${DATA}/model/${TARGET}_${DRAFT}_speculativedecoding_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}/ \
                --dataset $DATA --split test --temperature ${TEMPERATURE} \
                --draft_ensemble_weights ${STATIC_EW} \
                --num_assistant_tokens ${NUM_DRAFT} \
                --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPLES} \
                > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}.log 2>&1
            
            echo "Completed ${DATA} with target: ${TARGET}, draft: ${DRAFT}, draft length: ${NUM_DRAFT}"
        done
    done
done
