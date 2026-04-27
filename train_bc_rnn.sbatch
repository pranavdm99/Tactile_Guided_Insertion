#!/bin/bash
#SBATCH --job-name=train_bc_rnn
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=enpm690-class

# Load apptainer module
module load apptainer

# Image path
SIF_IMAGE="tactile_insertion.sif"

# Verify image exists
if [ ! -f "$SIF_IMAGE" ]; then
    echo "Error: $SIF_IMAGE not found. Please run the build job first."
    exit 1
fi

echo "[HPC] Starting BC-RNN Training Job..."
echo "[HPC] GPU: $(nvidia-smi -L)"

# Ensure real-time logging
export PYTHONUNBUFFERED=1

# Run training inside container
# We bind the current directory to /app
# We pass the WANDB_API_KEY from the environment
apptainer run --nv \
    --bind $PWD:/app \
    --pwd /app \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_MODE=offline \
    $SIF_IMAGE \
    python3 scripts/train_bc.py \
    --data datasets/data_round_nut_175d_260426 \
    --output runs/bc_rnn_$(date +%Y%m%d_%H%M%S) \
    --epochs 200 \
    --batch-size 16 \
    --num-workers 8 \
    --cache-to-ram \
    --wandb-project tactile-bc
