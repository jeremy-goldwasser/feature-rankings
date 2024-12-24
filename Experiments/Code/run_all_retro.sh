#!/bin/bash

methods=("ss kernelshap") # Add more methods if needed
datasets=("census" "bank" "brca" "breast_cancer" "credit")

for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        bash run_retrospective.sh "$method" "$dataset"
    done
done