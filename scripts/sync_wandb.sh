#!/bin/bash
# Script to manually sync offline wandb logs from a login node

echo "Starting W&B offline sync loop..."
echo "Press Ctrl+C to stop."

# Ensure we are in the project root
cd "$(dirname "$0")/.."

module load apptainer
SIF_IMAGE="tactile_insertion.sif"

while true; do
  # Find offline run directories. 
  OFFLINE_RUNS=$(find wandb -maxdepth 1 -name "offline-run-*" -type d 2>/dev/null)
  
  if [ -n "$OFFLINE_RUNS" ]; then
    echo "$(date): Found offline runs. Syncing via Apptainer..."
    # Use apptainer to run the sync command
    # Force 1 thread to avoid hitting login node process limits
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    apptainer exec --bind $PWD:/app --pwd /app $SIF_IMAGE wandb sync wandb/offline-run-*
  else
    echo "$(date): No offline runs found to sync."
  fi
  
  echo "Sleeping for 30 seconds..."
  sleep 30
done
