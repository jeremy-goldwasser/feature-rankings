#!/bin/bash
#SBATCH --job-name=shap_k5
#SBATCH -o shap_k5.out #File to which standard out will be written

mydir="$(dirname "$(readlink -f "$0")")"
python -u "$mydir/rankshap_nn.py" census 5