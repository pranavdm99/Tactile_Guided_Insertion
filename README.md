# 🦾 Tactile-Guided Insertion

**A high-fidelity tactile simulation pipeline built on Robosuite (Mujoco) and FOTS for insertion task.**

## 🖊️ Authors
- Pranav Deshakulkarni Manjunath: [pranavdeshakulkarni@gmail.com](mailto:pranavdeshakulkarni@gmail.com)
- Tirth Sadaria: [tsadaria@umd.edu](mailto:tsadaria@umd.edu)

> [!NOTE]
> This is a work in progress. We are actively developing this pipeline to enable tactile-based robotic insertion. 
---

## 📁 Project Structure
```
.
├── datasets/
├── env_setup/
│   ├── grippers/
│   │   ├── bringup/
│   │   │   └── test_fots_panda.py
│   │   ├── fots_panda.py
│   │   ├── __init__.py
│   │   ├── models/
│   │   │   ├── fots_panda_gripper.xml
│   │   │   └── meshes/
│   │   │       └── panda_gripper
│   │   │           ├── finger_longer.stl
│   │   │           ├── finger.stl
│   │   │           ├── finger_vis.stl
│   │   │           ├── hand.stl
│   │   │           └── hand_vis.stl
│   ├── __init__.py
│   ├── make_env.py
│   ├── tactile_depth_capture.py
│   ├── tactile_wrapper.py
│   └── utils/
│       └── data_recorder.py
├── scripts/
│   ├── teleop_keyboard_mouse.py
│   ├── validate_dataset.py
│   └── visualize_demo_data.py
├── docker-compose.yml
├── Dockerfile
├── docker_run.sh
├── entrypoint.sh
├── hydrate_fots_engine.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Quick Start

### 0. Clone the repository
```bash
git clone git@github.com:pranavdm99/Tactile_Guided_Insertion.git
cd Tactile_Guided_Insertion
```


### 1. Launch environment
- The [docker_run.sh](./docker_run.sh) script automatically builds and sets up the environment in a docker container.
  > [!Important]
  > Requires NVIDIA Container Toolkit to be installed on the host machine. [Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

  ```bash
    ./docker_run.sh           # Run from the project's root directory
  ```
- Attach to the container:
  ```bash
  docker compose exec tactile_insertion bash
  ```

### 2. Teleop and collect data
- Use the keyboard and mouse to perform the insertion task.
  ```bash
  python3 scripts/teleop_keyboard_mouse.py
  ```
  - **Controls**: 
    - Mouse: `Pan` (X-Y), `Wheel` (Z-axis)
    - Keyboard: `Arrow Keys` (Roll/Pitch), `PgUp/PgDn` (Yaw), `Enter` (Grasp).
    - **Save**: Press `R` to start or stop recording.
    - **Reset**: Press `Esc` to reset the environment.

### 3. Validate dataset with playback
Visualize your captured data with the live analytics panel.
```bash
python3 scripts/validate_dataset.py datasets/your_demo.hdf5 --play
```
- **Panel View**: 
  - **Left**: Agentview + Tactile images
  - **Right**: Analytics Panel showing Actions, Rewards, and Dones.

- **Detailed diagnostic plots**
  - Generate a full time-series report for an episode to verify control-signal synchronicity.
  ```bash
  python3 scripts/visualize_demo_data.py datasets/your_demo.hdf5 --demo demo_0
  ```
  - **Output**: Detailed `.png` report in `plots/` showing Actions vs. Tactile peak correlation.
---

## Acknowledgements
This project is built upon the following codebases:
- [Robosuite](https://github.com/ARISE-Initiative/robosuite) provides the simulation environment for the robot and the task.
- [FOTS-mujoco](https://github.com/Rancho-zhao/FOTS/tree/FOTS-mujoco) provides the tactile simulation engine for the gripper in MuJoCo.

We thank the authors for their contributions. 