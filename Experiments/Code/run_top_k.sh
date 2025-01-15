#!/bin/bash

# Ensure the top_K_outputs directory exists
mkdir -p top_K_outputs

if [ "$#" -ge 4 ]; then
    if [ "$#" -eq 5 ]; then
        slurm_file_name="top_K_outputs/${1}_${2}_${3}_${4}_${5}.out"
    else
        slurm_file_name="top_K_outputs/${1}_${2}_${3}_${4}.out"
    fi
    sbatch --output="$slurm_file_name" --wrap="python -u top_k_experiment.py --method=$1 --dataset=$2 --k=$3 --alpha=$4 ${5:+--guarantee=$5}"
else
    echo "Usage: $0 <method> <dataset> <k> <alpha> [guarantee]"
    exit 1
fi