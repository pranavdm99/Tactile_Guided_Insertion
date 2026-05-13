# ENPM690 Final Project — Tactile-Guided Robotic Insertion
### Group 5

**Team**
- Pranav Deshakulkarni Manjunath — pranavdeshakulkarni@gmail.com
- Tirth Sadaria — tsadaria@umd.edu

---

## Overview

This project implements a tactile-guided robotic peg-in-hole insertion task using:
- **FOTS (Fingertip OpTical Simulation)** — high-fidelity tactile sensor simulation on a Panda gripper
- **BC-RNN (Behavioral Cloning with LSTM)** — trained on 172 human teleoperation demonstrations
- **Robosuite + MuJoCo** — physics simulation environment

The robot must grasp a round nut and insert it onto a peg using tactile feedback from both fingertips. The policy observes proprioceptive state plus 64-dimensional low-dim tactile features (8×8 average-pooled, baseline-subtracted) from both fingers.

---

## Project Structure

```
group5/
├── Dockerfile                   # Lean inference container (CUDA 12.1 runtime)
├── docker-compose.yml
├── requirements.txt
├── entrypoint.sh
├── robomimic/                   # Custom robomimic fork (BC-RNN training framework)
│   └── robomimic/scripts/rollout_fots.py   ← primary inference entry point
├── robosuite/                   # Custom robosuite fork (FOTS gripper integration)
├── env_setup/                   # FOTSPandaGripper registration + environment factory
├── fots_sim/                    # FOTS tactile simulation MLP engine
├── scripts/                     # Teleoperation, evaluation, training utilities
├── configs/
│   └── round_nut_bc_rnn.json   # BC-RNN training configuration
├── checkpoints/
│   └── model_epoch_2800.pth    # Best trained policy (48 MB)
└── datasets/
    └── merged_lowdim.hdf5      # Preprocessed training dataset, 172 demos (33 MB)
```

---

## Prerequisites

- Docker Engine
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA GPU (compute capability ≥ 6.0)

---

## Quick Start — Run Inference

### 1. Build the Docker image

```bash
docker compose build
```

> First build takes ~10–15 minutes (downloads CUDA base + installs PyTorch, MuJoCo, robosuite).

### 2. Run the pre-trained model (10 rollouts)

```bash
docker compose run --rm tactile_inference
```

This runs 10 rollouts with `checkpoints/model_epoch_2800.pth` and saves a video to `output/rollout.mp4`.

### 3. Custom run options

```bash
# Change number of rollouts or horizon
docker compose run --rm tactile_inference \
  python3 /app/robomimic/robomimic/scripts/rollout_fots.py \
  --agent   /app/checkpoints/model_epoch_2800.pth \
  --n_rollouts 5 \
  --horizon    400 \
  --video_path /app/output/my_rollout.mp4

# Drop into an interactive shell
docker compose run --rm --entrypoint bash tactile_inference
```

The rollout script prints per-attempt stats and a final success-rate summary:
```
--- Rollout 1/10 ---
{"Return": 46.9, "Horizon": 400, "Success": 0.0}
...
=== Average Rollout Stats ===
{
    "Return": 44.2,
    "Horizon": 400.0,
    "Num_Success": 1
}
```

---

## Results

| Checkpoint | Eval Rollouts | Successes | Success Rate |
|------------|--------------|-----------|--------------|
| Epoch 2800 | 200          | 20         | 10%       |

The task requires millimeter-precision peg-in-hole insertion using tactile sensing alone — even a low success rate represents meaningful policy learning from BC on a high-tolerance manipulation task.

A highlight compilation video of successful insertions is available at: **[TODO: add Google Drive / OneDrive link]**

---

## Training Reference

> The Docker container is inference-only. Training was done on Zaratan HPC (A100 GPU, ~18 hours for 3000 epochs).

### Step 1 — Collect demonstrations (teleoperation)

```bash
docker compose run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --entrypoint bash tactile_inference \
  -c "python3 /app/scripts/teleop_keyboard_mouse.py"
```

**Controls**
| Input | Action |
|-------|--------|
| Mouse move | X-Y translation |
| Mouse wheel | Z translation |
| Arrow keys | Roll / Pitch |
| PgUp / PgDn | Yaw |
| Enter | Toggle gripper |
| R | Start / stop recording |
| Esc | Reset environment |

Demonstrations are saved as HDF5 files in `recorded_data/`.

### Step 2 — Preprocess tactile observations

Converts raw tactile images (96×128×3) to 64-dim low-dim features used by the policy:

```bash
python3 robomimic/preprocess_tactile_lowdim.py \
  --input  datasets/your_demos.hdf5 \
  --output datasets/merged_lowdim.hdf5
```

### Step 3 — Train BC-RNN policy

```bash
cd robomimic
python robomimic/scripts/train.py --config round_nut_bc_rnn.json
```

Training config highlights (`configs/round_nut_bc_rnn.json`):
- Algorithm: BC with GMM action head (5 modes)
- RNN: LSTM, 2 layers, hidden dim 1000
- Sequence length: 10 steps
- Epochs: 3000, batch size 100, lr 1e-4 (decays at epoch 2000)

### Dataset download

The raw HDF5 demonstration files (~1.2 GB total, 6 episodes) are available at:
**[TODO: add Google Drive / OneDrive link]**

The preprocessed `merged_lowdim.hdf5` (33 MB, 172 episodes) is included in `datasets/`.

---

## Architecture

```
Observations (9 keys)               Policy (BC-RNN)
─────────────────────               ───────────────
robot0_eef_pos        (3)  ──┐
robot0_eef_quat       (4)  ──┤      LSTM (2 layers, h=1000)
robot0_gripper_qpos   (2)  ──┤  ──►        │
object_pos            (3)  ──┤             ▼
object_quat           (4)  ──┤      GMM Head (5 modes)
object_to_eef_pos     (3)  ──┤             │
object_to_eef_quat    (4)  ──┤             ▼
tactile_left_lowdim  (64)  ──┤      Action (7-DoF)
tactile_right_lowdim (64)  ──┘      EEF delta pos/ori + gripper
```

Tactile features: raw fingertip image → baseline subtraction → grayscale → 8×8 average pool → flatten → /128 normalise → 64-dim vector.

---

## Acknowledgements

- [Robosuite](https://github.com/ARISE-Initiative/robosuite) — robot simulation framework
- [robomimic](https://github.com/ARISE-Initiative/robomimic) — imitation learning framework
- [FOTS-mujoco](https://github.com/Rancho-zhao/FOTS/tree/FOTS-mujoco) — tactile sensor simulation
