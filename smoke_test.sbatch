#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --account=enpm690-class

module load apptainer

SIF_IMAGE="tactile_insertion.sif"

if [ ! -f "$SIF_IMAGE" ]; then
    echo "Error: $SIF_IMAGE not found."
    exit 1
fi

echo "[HPC] Running Smoke Test..."
apptainer exec --nv \
    --bind $PWD:/app \
    --pwd /app \
    $SIF_IMAGE \
    python3 scripts/test_env_hpc.py
