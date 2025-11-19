#!/usr/bin/env bash

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --gpus=1
#SBATCH --qos=dw87

###########SBATCH --partition=m13l

# Fail on any error
set -euo pipefail

cd /home/jcdutoit/Research/nn_gad
uv run main.py