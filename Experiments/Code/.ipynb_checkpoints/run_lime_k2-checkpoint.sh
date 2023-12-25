#!/bin/bash
#SBATCH --job-name=lime_k2
#SBATCH -o lime_k2.out #File to which standard out will be written

python -u "lime_rf.py" 2
