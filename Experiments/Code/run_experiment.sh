#!/bin/bash
#SBATCH --job-name=${1}_${2}_k${3}_${4} # e.g. shap_census_k2_0.10
#SBATCH -o ${1}_${2}_k${3}_${4}.out # File to which standard out will be written

python -u "experiment.py" --method=$1 --dataset=$2 --k=$3 --alpha=$4
