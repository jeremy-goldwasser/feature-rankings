#!/bin/bash

if [ "$#" -ge 4 ]; then
    if [ "$#" -eq 5 ]; then
        slurm_file_name="${1}_${2}_${3}_${4}_${5}.out"
        python -u "top_k_experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4 --guarantee=$5 > "$slurm_file_name"
    else
        slurm_file_name="${1}_${2}_${3}_${4}.out"
        python -u "top_k_experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4 > "$slurm_file_name"
    fi
else
    echo "Usage: $0 <method> <dataset> <k> <alpha> [guarantee]"
    exit 1
fi