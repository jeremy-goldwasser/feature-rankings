#!/bin/bash
#SBATCH --job-name=lime_k5
#SBATCH -o lime_k5.out #File to which standard out will be written

python -u "lime_rf.py" 5
