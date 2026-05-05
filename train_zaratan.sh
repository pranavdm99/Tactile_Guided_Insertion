#!/bin/bash
#SBATCH --job-name=bc_rnn_fots
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=enpm690-class
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tsadaria@umd.edu

# ── This script must be submitted from inside the robomimic folder ─────── #
#   cd ~/scratch/robomimic
#   sbatch train_zaratan.sh

set -e

SUBMIT_DIR="${SLURM_SUBMIT_DIR}"          # directory sbatch was called from
DATASET="${SUBMIT_DIR}/datasets/merged_lowdim.hdf5"
OUTPUT_DIR="${SUBMIT_DIR}/trained_models/round_nut"
TEMP_CONFIG="/tmp/bc_rnn_${SLURM_JOB_ID}.json"

# ── Modules ────────────────────────────────────────────────────────────── #
module purge
module load anaconda

# ── Activate conda env ─────────────────────────────────────────────────── #
# Change "robomimic_venv" to whatever your conda env is named on Zaratan
conda activate robomimic_venv

# ── Sanity checks ──────────────────────────────────────────────────────── #
echo "========================================"
echo "Job ID   : ${SLURM_JOB_ID}"
echo "Node     : ${SLURMD_NODENAME}"
echo "GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Submit   : ${SUBMIT_DIR}"
echo "Dataset  : ${DATASET}"
echo "Output   : ${OUTPUT_DIR}"
echo "Start    : $(date)"
echo "========================================"

if [ ! -f "${DATASET}" ]; then
    echo "ERROR: dataset not found at ${DATASET}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p /tmp/robomimic_logs

export PYTHONUNBUFFERED=1
export WANDB_API_KEY="${WANDB_API_KEY}"   # set this on login node: export WANDB_API_KEY=<your_key>

# ── Stamp real paths into a temp copy of the config ───────────────────── #
sed "s|DATASET_PATH|${DATASET}|g; s|OUTPUT_DIR|${OUTPUT_DIR}|g" \
    "${SUBMIT_DIR}/round_nut_bc_rnn.json" > "${TEMP_CONFIG}"

echo "Config written to ${TEMP_CONFIG}"

# ── Run training ───────────────────────────────────────────────────────── #
cd "${SUBMIT_DIR}"
python robomimic/scripts/train.py --config "${TEMP_CONFIG}"

echo "========================================"
echo "End : $(date)"
echo "Done."
echo "========================================"
