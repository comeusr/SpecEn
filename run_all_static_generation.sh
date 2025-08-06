#!/bin/bash

for weight in $(seq 0.6 0.1 1.0); do
    ./generation.sh "$weight"
done