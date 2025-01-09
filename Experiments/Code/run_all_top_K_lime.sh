#!/bin/bash

methods=("lime")
datasets=("census" "bank" "brca" "breast_cancer" "credit")
ks=(2 5)
alphas=(0.1 0.2)

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        for k in "${ks[@]}"; do
            for alpha in "${alphas[@]}"; do
                sbatch run_top_k.sh "$method" "$dataset" "$k" "$alpha"
            done
        done
    done
done