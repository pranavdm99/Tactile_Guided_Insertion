#!/bin/bash
set -e

echo "[ENPM690 Group 5] Tactile-Guided Insertion — Inference Container"
echo "  Model   : /app/checkpoints/model_epoch_2800.pth"
echo "  Output  : /app/output/"
echo ""

exec "$@"
