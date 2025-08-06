#!/bin/bash

for ITER in $(seq 32 32 384); do
    ./generation.sh "$ITER"
done