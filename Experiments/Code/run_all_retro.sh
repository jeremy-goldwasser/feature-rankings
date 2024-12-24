#!/bin/bash

# Ensure the output directory exists
mkdir -p retro_outputs

# Parameters
methods=("ss kernelshap") 
datasets=("census" "bank" "brca" "breast_cancer" "credit")

# Loop through all combinations
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        sbatch --output="retro_outputs/%x_%j.out" \
               --error="retro_outputs/%x_%j.err" \
               retrospective_experiment.sh "$method" "$dataset"
    done
done