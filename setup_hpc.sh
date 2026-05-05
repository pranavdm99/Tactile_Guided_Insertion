#!/bin/bash
# One-time setup on Zaratan — run this ONCE from the login node before submitting jobs.
# Usage:
#   cd ~/scratch/robomimic
#   bash setup_hpc.sh

set -e

echo "=== Loading modules ==="
module purge
module load anaconda

echo "=== Creating conda env: robomimic_venv (Python 3.8) ==="
conda create -n robomimic_venv python=3.8 -y

echo "=== Activating env ==="
source activate robomimic_venv

echo "=== Installing PyTorch (CUDA 11.8) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "=== Installing training dependencies ==="
pip install -r requirements_train.txt

echo "=== Installing robomimic in dev mode ==="
pip install -e .

echo ""
echo "=== Setup complete! ==="
echo "To submit training: cd ~/scratch/robomimic && sbatch train_zaratan.sh"
