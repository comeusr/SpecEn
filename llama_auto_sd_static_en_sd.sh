#!/bin/bash


# vanilla autoregressive decoding
# Datasets to loop over
DATASETS=("cnndm" "wmt" "xsum")

# Gemma models to loop over
# MODELS=("gemma-3-4b-it" "gemma-3-12b-it" "gemma-3-1b-it")
MODELS=("Llama-3.1-8B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.2-1B-Instruct")

# Fixed parameters
METHOD=auto
# CLASS=google
CLASS=meta-llama
STATIC_EW=0.0
TEMPERATURE=0.2
ITER=264
NUM_DRAFT=5
N_EXAMPLES=200
DO_SAMPLE=False

export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
export HF_TOKEN=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
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Loop over each dataset and model combination
for DATA in "${DATASETS[@]}"; do
    mkdir -p .logs/$DATA
    
    for TARGET in "${MODELS[@]}"; do
        echo "Processing dataset: $DATA with model: $TARGET"
        
        python speculative_decoding.py \
            --method ${METHOD} --do_sample ${DO_SAMPLE} \
            --draft_model ${CLASS}/${TARGET} \
            --target_model ${CLASS}/${TARGET} \
            --model_path ../data/${DATA}/model/${TARGET}_autoregressivedecoding_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}/ \
            --dataset $DATA --split test --temperature ${TEMPERATURE} \
            --draft_ensemble_weights ${STATIC_EW} \
            --num_assistant_tokens ${NUM_DRAFT} \
            --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPLES} \
            > .logs/${DATA}/${TARGET}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}.log 2>&1
        
        echo "Completed autoregressive ${DATA} with ${TARGET}"
    done
done





# naive speculative decoding

# Datasets to loop over
DATASETS=("cnndm" "wmt" "xsum")

# Model combinations (target, draft)
declare -A MODEL_PAIRS=(
    ["Llama-3.1-8B-Instruct"]="Llama-3.2-1B-Instruct"
    ["Llama-3.1-8B-Instruct"]="Llama-3.2-3B-Instruct"
    ["Llama-3.2-3B-Instruct"]="Llama-3.2-1B-Instruct"
)

# Draft lengths to try
DRAFT_LENGTHS=(3 4 5 6 7 8 9 10 11 12 13 14 15)

# Fixed parameters
METHOD=sd
CLASS=meta-llama
STATIC_EW=0.0
TEMPERATURE=0.2
ITER=264
N_EXAMPLES=200
DO_SAMPLE=False

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
                --model_path ../data/${DATA}/model/${TARGET}_${DRAFT}_speculativedecoding_DRAFT_LEN${NUM_DRAFT}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}/ \
                --dataset $DATA --split test --temperature ${TEMPERATURE} \
                --draft_ensemble_weights ${STATIC_EW} \
                --num_assistant_tokens ${NUM_DRAFT} \
                --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPLES} \
                > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}.log 2>&1
            
            echo "Completed ${DATA} with target: ${TARGET}, draft: ${DRAFT}, draft length: ${NUM_DRAFT}"
        done
    done
done





# static ensemble
# Datasets to loop over
DATASETS=("cnndm" "wmt" "xsum")

# Model combinations (target, draft)
declare -A MODEL_PAIRS=(
    ["Llama-3.1-8B-Instruct"]="Llama-3.2-1B-Instruct"
    ["Llama-3.1-8B-Instruct"]="Llama-3.2-3B-Instruct"
    ["Llama-3.2-3B-Instruct"]="Llama-3.2-1B-Instruct"
)

# Draft lengths to try
DRAFT_LENGTHS=(3 4 5 6 7 8 9 10 11 12 13 14 15)

# Static ensemble weights to try
STATIC_EWS=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

# Fixed parameters
METHOD=static_en
CLASS=meta-llama
TEMPERATURE=0.2
ITER=264
N_EXAMPLES=200
DO_SAMPLE=False

export WANDB_API_KEY=c06454b9d39ecbc38415f676534da6704a3050c0
export HF_TOKEN=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
export TRANSFORMERS_VERBOSITY=error

wandb login $WANDB_API_KEY
huggingface-cli login --token $HF_TOKEN

# Loop over each dataset, model combination, draft length, and static ensemble weight
for DATA in "${DATASETS[@]}"; do
    mkdir -p .logs/$DATA
    
    for TARGET in "${!MODEL_PAIRS[@]}"; do
        DRAFT=${MODEL_PAIRS[$TARGET]}
        
        for NUM_DRAFT in "${DRAFT_LENGTHS[@]}"; do
            for STATIC_EW in "${STATIC_EWS[@]}"; do
                echo "Processing dataset: $DATA with target: $TARGET, draft: $DRAFT, draft length: $NUM_DRAFT, static_ew: $STATIC_EW"
                
                python speculative_decoding.py \
                    --method ${METHOD} --do_sample ${DO_SAMPLE} \
                    --draft_model ${CLASS}/${DRAFT} \
                    --target_model ${CLASS}/${TARGET} \
                    --model_path ../data/${DATA}/model/${TARGET}_${DRAFT}_staticensemble_DRAFTWEIGHT${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}/ \
                    --dataset $DATA --split test --temperature ${TEMPERATURE} \
                    --draft_ensemble_weights ${STATIC_EW} \
                    --num_assistant_tokens ${NUM_DRAFT} \
                    --max_tokens 128 --batch_size 1 --n_examples ${N_EXAMPLES} \
                    > .logs/${DATA}/${TARGET}_${DRAFT}_${METHOD}_${STATIC_EW}_DRAFT_LEN${NUM_DRAFT}_${ITER}_NUM_EXMPLS${N_EXAMPLES}_TEMP${TEMPERATURE}_DO_SAMPLe${DO_SAMPLE}.log 2>&1
                
                echo "Completed static ensemble with ${DATA} with target: ${TARGET}, draft: ${DRAFT}, draft length: ${NUM_DRAFT}, static_ew: ${STATIC_EW}"
            done
        done
    done
done

