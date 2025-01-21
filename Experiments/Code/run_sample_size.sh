#!/bin/bash

slurm_file_name="sample_size.out"
sbatch --output="$slurm_file_name" --wrap="python -u sample_size_experiment.py"
