# Tactile-Guided Insertion

**Tactile-Guided Insertion using FOTS and robosuite for peg-in-hole task. The system is fully calibrated and verified for large-scale data collection**
---

### Core Specifications
- **Tactile Resolution**: Dual 320x240 RGB "DIGIT" imprints.
- **Physics**: "Zero-Resistance" soft-contact model with 50,000 KP actuators.
---

## 📁 Project Structure
```
├── fots_sim/                   # Core Tactile Simulation (MLP & Assets)
├── env_setup/                  # Robosuite Integration & Sandboxes
│   ├── custom_grippers.py     # Master FOTSPandaGripper definition
│   ├── test_franka_gripper_fots.py  # Interactive Franka Sandbox
│   ├── test_fots_primitives.py      # Standalone Primitive Sandbox
│   ├── tactile_depth_capture.py     # Depth capture utility
│   └── tactile_wrapper.py     # Robosuite Observation Wrapper
├── scripts/                    # Data Collection & Playback
└── README.md                   # This master document
```
---

## Quick Start

### 1. Build and Run the Docker Container
Build the container using the following command:
```bash
docker compose up -d --build
docker compose exec tactile_insertion bash
```
### 2. Launch Interactive Sandbox
Verify the tactile synthesis and object manipulation in real-time.
```bash
docker exec tactile_insertion env PYTHONPATH=/app python3 /app/env_setup/test_franka_gripper_fots.py --shape box
```
> **Tip**: Drive the **SQUEEZE** slider past 100 to see high-contrast tactile imprints.

### 2. Verify Primitive Shapes
Test tactile response on Sphere, Box, and Cylinder primitives.
```bash
docker exec tactile_insertion env PYTHONPATH=/app python3 /app/env_setup/test_fots_primitives.py --shape sphere
```

## 🤖 Automated FOTS Setup
The repository is designed to be **Self-Hydrating**. When you launch the Docker container, the following happens automatically:
1.  **Detection**: The `entrypoint.sh` checks for the existence of `FOTS_repo`.
2.  **Auto-Clone**: If missing, it uses `git` to clone the target research repository from `https://github.com/Rancho-zhao/FOTS` on branch `FOTS-mujoco`.
3.  **Extraction**: It runs `scripts/hydrate_fots_engine.py` to surgically extract weights and logic into `fots_sim`.
4.  **Standalone Mode**: Once hydrated, the container uses the standalone `fots_sim` engine.

To manually trigger a sync or checkout a specific branch:
```bash
python3 scripts/hydrate_fots_engine.py --url <repo_url> --branch <branch_name>
```

---
## 🔧 Performance & Toggles
- **Disable Shadows**: Edit `fots_sim/utils/mlp_render.py` → `generate(shadow=False)` for ~30% FPS boost.
- **CPU Mode**: In `env_setup/tactile_wrapper.py`, set `device = torch.device('cpu')`.
---
