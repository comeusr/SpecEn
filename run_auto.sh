#!/bin/bash

# Datasets to loop over
DATASETS=("cnndm" "wmt" "xsum")

# Gemma models to loop over
# MODELS=("gemma-3-4b-it" "gemma-3-12b-it" "gemma-3-1b-it")
MODELS=("Llama-3.2-8B-Instruct" "Llama-3.2-3B-Instruct" "Llama-3.2-1B-Instruct")

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
        
        echo "Completed ${DATA} with ${TARGET}"
    done
done

