#!/bin/bash
set -e

echo "[FOTS BOOTSTRAP] Initializing Tactile Simulation Environment..."

# 1. Automatic Hydration & Sync
# The script will:
# - Clone FOTS_repo if missing (from the official fork)
# - Extract relevant parts into fots_sim
# - Resume if fots_sim is already self-contained
python3 /app/hydrate_fots_engine.py

# 2. Execute Command (default: bash)
exec "$@"
