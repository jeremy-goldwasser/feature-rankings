#!/bin/bash
#SBATCH --job-name=k2
#SBATCH -o k2.out #File to which standard out will be written
#SBATCH -e k2.err #File to which standard err will be written
python -u /accounts/grad/jeremy_goldwasser/RankSHAP/Experiments/Code/rankshap_nn.py census 2
