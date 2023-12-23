#!/bin/bash
#SBATCH --job-name=lime_k2
#SBATCH -o lime_k2.out #File to which standard out will be written

mydir="$(dirname "$(readlink -f "$0")")"
python -u "$mydir/lime_rf.py" 2
