#!/bin/bash
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1_$2
#SBATCH -o $1_$2.out #e.g. lime_brca.out
EOT
python -u "experiment.py" --method=$1 --dataset=$2
