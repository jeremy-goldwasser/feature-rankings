#!/bin/bash
#SBATCH --job-name=lime_k5
#SBATCH -o lime_k5.out #File to which standard out will be written
#SBATCH -e lime_k5.err #File to which standard err will be written
python -u /accounts/grad/jeremy_goldwasser/RankSHAP/Experiments/Code/lime_rf.py 2
