#!/bin/bash

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

accelerate launch \
    --config_file accelerate_config/fsdp_8gpu.yaml \
    launch.py loss=sft model=llama datasets=[gsm8k] exp_name=llama3-8B-sft \