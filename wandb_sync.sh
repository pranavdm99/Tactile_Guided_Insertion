#!/bin/bash
# Run this on the Zaratan LOGIN NODE after any training job finishes.
# It syncs all offline wandb runs in this folder to wandb.ai.
#
#   cd ~/scratch/robomimic
#   bash wandb_sync.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANDB_DIR="${SCRIPT_DIR}/wandb"

source /scratch/zt1/project/enpm690/user/tsadaria/miniconda3/etc/profile.d/conda.sh
conda activate robomimic_venv

echo "Syncing all offline wandb runs in ${WANDB_DIR} ..."
wandb sync --sync-all "${WANDB_DIR}"
echo "Done."
