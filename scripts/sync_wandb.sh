#!/bin/bash

SIF_IMAGE="tactile_insertion.sif"
SYNC_DIR="wandb"
SLEEP_INTERVAL=30

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

trap "echo -e '\n[SYSTEM] Received stop signal. Exiting gracefully...'; exit 0" SIGINT SIGTERM

echo "==================================================="
echo " Starting W&B offline sync loop "
echo " Detach from tmux: Ctrl+B, then D"
echo " Stop the script:  Ctrl+C"
echo "==================================================="

cd "$(dirname "$0")/.." || { echo "Failed to change directory"; exit 1; }

module load apptainer 2>/dev/null || echo "[Warning] module load apptainer failed or not needed."

while true; do
    echo "------------------------------------------------"
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ ! -d "$SYNC_DIR" ]; then
        echo "[$TIMESTAMP] No '$SYNC_DIR' directory found. Sleeping for ${SLEEP_INTERVAL}s..."
        sleep "$SLEEP_INTERVAL"
        continue
    fi

    # Find directories matching offline-run-* inside wandb
    OFFLINE_COUNT=$(find "$SYNC_DIR" -maxdepth 1 -name "offline-run-*" -type d 2>/dev/null | wc -l)
    
    if [ "$OFFLINE_COUNT" -eq 0 ]; then
        echo "[$TIMESTAMP] Directory exists, but 0 offline runs found. Sleeping for ${SLEEP_INTERVAL}s..."
    else
        echo "[$TIMESTAMP] Found $OFFLINE_COUNT offline run(s). Spawning Apptainer for sync..."
        
        # Let bash expand the wildcard here outside quotes so it passes all matching dirs
        nice -n 19 apptainer exec \
            --bind "$PWD":/app \
            --pwd /app \
            "$SIF_IMAGE" \
            bash -c 'wandb sync wandb/offline-run-*'

        SYNC_STATUS=$?
        if [ $SYNC_STATUS -eq 0 ]; then
            echo "[$TIMESTAMP] Sync cycle completed successfully."
        else
            echo "[$TIMESTAMP] Sync cycle exited with code $SYNC_STATUS. Will retry later."
        fi
    fi

    sleep "$SLEEP_INTERVAL"
done