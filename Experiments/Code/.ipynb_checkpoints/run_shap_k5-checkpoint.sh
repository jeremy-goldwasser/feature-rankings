#!/bin/bash
#SBATCH --job-name=shap_k5
#SBATCH -o shap_k5.out #File to which standard out will be written

python -u "rankshap_nn.py" census 5