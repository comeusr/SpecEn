#!/bin/bash

# Nested loop: gamma from 5 to 25 (step 5), val from 0.1 to 1.0 (step 0.1)
for gamma in $(seq 5 5 25); do
    for val in $(seq 0.0 0.1 1.0); do
        ./run_sd.sh 2,3 2 cnndm Qwen3-14B Qwen3-0.6B "$gamma" constant "$val" 0.5 256
    done
done