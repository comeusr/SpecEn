#!/bin/bash

for lambda in $(seq 2 2 8); do
    ./run_rl.sh "$lambda"
done