#!/bin/bash
#SBATCH --job-name=small
#SBATCH --output=small.out
#SBATCH --error=small.err
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load midway2; module load python/3.7.0
pip install aequitas --user
python3 main.py
