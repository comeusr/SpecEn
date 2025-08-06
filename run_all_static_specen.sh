#!/bin/bash

for weight in $(seq 0.0 0.1 1.0); do
    ./run_specenhead.sh "$weight"
done