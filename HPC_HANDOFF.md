# HPC Handoff — robomimic BC-RNN FOTS Training

This file gives you full context to continue this project. Read it entirely before doing anything.

---

## What This Project Is

Train a BC-RNN (Behavior Cloning + RNN) policy using the `robomimic` framework to perform
**round nut assembly** on a Panda robot. Demonstrations were collected using a custom FOTS
tactile gripper (`FOTSPandaGripper`) with tactile sensor images. The tactile images were
converted to 64-dim low-dim features to keep training fast.

The goal is to get a trained policy checkpoint running on Zaratan HPC (UMD).

---

## Repo / Folder Structure

Everything needed is inside one folder that was SCP'd to Zaratan:

```
~/scratch/robomimic/                        ← submit sbatch from here
├── datasets/
│   └── merged_lowdim.hdf5                 ← 172 demos, 33MB, ready for training
├── round_nut_bc_rnn.json                   ← training config (DATASET_PATH / OUTPUT_DIR are placeholders)
├── train_zaratan.sh                        ← SLURM sbatch script, ready to submit
├── setup_hpc.sh                            ← one-time conda env setup script
├── preprocess_tactile_lowdim.py            ← already ran locally, do NOT re-run
└── robomimic/                              ← source code with all bug fixes applied
    ├── utils/dataset.py
    ├── utils/train_utils.py
    ├── envs/env_robosuite.py
    └── scripts/run_trained_agent.py
```

---

## Bug Fixes Already Applied (do NOT revert)

| File | What was changed |
|---|---|
| `robomimic/utils/dataset.py:516` | `np.bool` → `np.bool_` (NumPy deprecation) |
| `robomimic/utils/train_utils.py:141` | `load_next_obs=True` → `load_next_obs=False` (HDF5 has no next_obs group) |
| `robomimic/envs/env_robosuite.py:12` | `postprocess_model_xml` import wrapped in try/except with identity fallback |
| `robomimic/envs/env_robosuite.py:56` | Version check changed from `== "2"` to `>= 2` (int) to support robosuite 1.5.x |
| `robomimic/envs/env_robosuite.py:~179` | `env_name` popped from `env_kwargs` before unpacking (prevented duplicate kwarg crash) |
| `robomimic/envs/env_robosuite.py:~205` | Added `RoundNut_* / SquareNut_*` → `object_*` key remapping in `get_observation()` |
| `robomimic/scripts/run_trained_agent.py` | Added `sys.path.insert` + `import env_setup` at top to register `FOTSPandaGripper` |

---

## Training Config (`round_nut_bc_rnn.json`)

Key settings:
```
algo:           BC-RNN with GMM head
rnn:            LSTM, hidden_dim=1000, num_layers=2, horizon=10
gmm:            enabled, num_modes=5
epochs:         2000
steps/epoch:    500
batch_size:     100
rollout:        DISABLED (no live sim needed during training)
observations:
  - robot0_eef_pos        (3)
  - robot0_eef_quat       (4)
  - robot0_gripper_qpos   (2)
  - object_pos            (3)
  - object_quat           (4)
  - object_to_robot0_eef_pos   (3)
  - object_to_robot0_eef_quat  (4)
  - tactile_left_lowdim   (64)   ← converted from 96x128x3 tactile images
  - tactile_right_lowdim  (64)   ← converted from 96x128x3 tactile images
```

The config has `DATASET_PATH` and `OUTPUT_DIR` as string placeholders.
The sbatch script fills these in at runtime via `sed`. Do not hardcode paths in the JSON.

---

## Dataset: `merged_lowdim.hdf5`

- Merged from 6 original FOTS HDF5 demo files (172 total episodes)
- No image data — lean and fast to cache entirely in RAM
- Tactile images (96×128×3 uint8) were converted to 64-dim vectors per sensor:
  - Baseline = mean of first 5 frames per episode (robot not touching anything)
  - Deviation from baseline → grayscale → 8×8 average pool → divide by 128.0
- HDF5 obs keys: `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`,
  `object_pos`, `object_quat`, `object_to_robot0_eef_pos`, `object_to_robot0_eef_quat`,
  `tactile_left_lowdim`, `tactile_right_lowdim`

---

## Zaratan HPC — SLURM Config

```
#SBATCH --account=enpm690-class
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
```

Conda env name: `robomimic_venv` (Python 3.8)
User email: `tsadaria@umd.edu`

---

## What Still Needs to Be Done (in order)

### 1. Set up conda env on Zaratan (login node, one-time)

```bash
cd ~/scratch/robomimic
bash setup_hpc.sh
```

If `setup_hpc.sh` fails, run manually:
```bash
module purge
module load anaconda
conda create -n robomimic_venv python=3.8 -y
conda activate robomimic_venv
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install h5py numpy tqdm imageio tensorboard
pip install -e .
```

> NOTE: robosuite is NOT needed for training (rollouts are disabled).
> It is only needed for policy rollout evaluation after training.

### 2. Confirm scratch path

On Zaratan, run:
```bash
echo $SCRATCH
```
The actual scratch path is usually `/scratch/zt1/project/enpm690/tsadaria/`.
The sbatch uses `$SLURM_SUBMIT_DIR` so paths resolve automatically as long as
you `cd` into the robomimic folder before calling `sbatch`.

### 3. Submit training

```bash
cd ~/scratch/robomimic          # or wherever you SCP'd it
sbatch train_zaratan.sh
```

Monitor with:
```bash
squeue -u tsadaria
tail -f train_<JOBID>.out
```

### 4. After training — rollout evaluation (known issue)

`run_trained_agent.py` will fail to generate policy rollout videos because:
- The policy was trained with custom observations (`object_pos` etc.) that are renamed
  from robosuite's native `RoundNut_pos` by the `TactileObservationWrapper`
- The rollout env does not apply this wrapper, so the policy crashes on missing obs keys
- The `FOTSPandaGripper` must also be registered before env creation

To fix rollout evaluation, the env creation in `run_trained_agent.py` needs to:
1. Import and register `FOTSPandaGripper` from `Tactile_Guided_Insertion/env_setup/grippers/`
2. Wrap the env with `TactileObservationWrapper` (or at minimum remap obs keys)

This is NOT blocking for training — only needed for generating policy videos after training.

---

## Key Paths (local machine, for reference)

```
Local project root:    /media/tirth/Expansion/docker_data_mount/projects/enpm690/
robomimic:             .../enpm690/robomimic/
Tactile_Guided_Insertion: .../enpm690/Tactile_Guided_Insertion/
Custom robosuite fork: .../enpm690/robosuite/   (has FOTSPandaGripper)
```

The `Tactile_Guided_Insertion` project is already on Zaratan but runs via apptainer
(Singularity container) — its robosuite installation is inside the container and NOT
directly pip-accessible on the host filesystem.
