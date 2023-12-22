#!/bin/bash
#SBATCH --job-name=shap_k5
#SBATCH -o shap_k5.out #File to which standard out will be written
#SBATCH -e shap_k5.err #File to which standard err will be written
python -u /accounts/grad/jeremy_goldwasser/RankSHAP/Experiments/Code/rankshap_nn.py census 5
