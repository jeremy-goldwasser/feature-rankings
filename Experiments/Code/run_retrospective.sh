#!/bin/bash

# Ensure the outputs directory exists
mkdir -p retro_outputs

if [ "$#" -eq 2 ]; then
    slurm_file_name="retro_outputs/retrospective_${1}_${2}.out"
    sbatch --output="$slurm_file_name" --wrap="python -u retrospective_experiment.py --method=$1 --dataset=$2"
else
    echo "Usage: $0 <method> <dataset>"
    exit 1
fi