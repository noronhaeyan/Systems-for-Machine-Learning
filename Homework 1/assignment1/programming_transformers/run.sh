#!/bin/bash

dataset=lm_synthetic 
epochs=50 
lr=0.0005
train_batch=32
num_layers=4  # Default value for num_layers

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        num_layers=*) num_layers="${1#*=}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# Check if python or python3 is available
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Python is not installed."
    exit 1
fi

$PYTHON_CMD src/experiments.py \
    --task ${dataset} \
    --epochs ${epochs} \
    --learning_rate ${lr} \
    --train_batch ${train_batch} \
    --layers ${num_layers}