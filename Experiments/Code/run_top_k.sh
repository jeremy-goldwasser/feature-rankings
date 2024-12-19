#!/bin/bash

if [ "$#" -ge 4 ]; then
    if [ "$#" -eq 5 ]; then
        python -u "top_k_experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4 --guarantee=$5
    else
        python -u "top_k_experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4
    fi
else
    echo "Usage: $0 <method> <dataset> <k> <alpha> [guarantee]"
    exit 1
fi