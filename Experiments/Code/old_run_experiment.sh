#!/bin/bash
#SBATCH --job-name="$1_$2_k$3_$4" # e.g. shap_census_k2_0.10
#SBATCH -o "$1_$2_k$3_$4.out" #File to which standard out will be written

python -u "experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4
