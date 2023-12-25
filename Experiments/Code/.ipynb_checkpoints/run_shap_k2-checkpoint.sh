#!/bin/bash
#SBATCH --job-name=shap_k2
#SBATCH -o shap_k2.out #File to which standard out will be written

python -u "rankshap_nn.py" census 2
