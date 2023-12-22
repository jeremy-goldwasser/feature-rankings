#!/bin/bash
#SBATCH --job-name=lime_k2
#SBATCH -o lime_k2.out #File to which standard out will be written
#SBATCH -e lime_k2.err #File to which standard err will be written
python -u /accounts/grad/jeremy_goldwasser/RankSHAP/Experiments/Code/lime_rf.py 2
