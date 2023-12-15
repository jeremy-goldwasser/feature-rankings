#!/bin/bash
#SBATCH --job-name=k5
#SBATCH -o k5.out #File to which standard out will be written
#SBATCH -e k5.err #File to which standard err will be written
python -u /accounts/grad/jeremy_goldwasser/RankSHAP/Experiments/Code/rankshap_nn.py census 5
