"""
Rollout evaluation for the FOTS BC-RNN policy trained with robomimic.

Bypasses robomimic's env_from_checkpoint to correctly set up:
  - FOTSPandaGripper registration
  - TactileObservationWrapper (96x128 tactile cameras, matching training data)
  - Online tactile-to-lowdim conversion (baseline-subtracted, 8x8 pooled, /128)
  - Object obs key remapping (RoundNut_* -> object_*)

Usage:
    conda activate robomimic_venv
    cd /media/tirth/Expansion/docker_data_mount/projects/enpm690/robomimic
    python robomimic/scripts/rollout_fots.py \
        --agent runs/model_epoch_2000.pth \
        --n_rollouts 10 --horizon 400 \
        --video_path runs/fots_rollout.mp4
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import imageio

TACTILE_ROOT = "/media/tirth/Expansion/docker_data_mount/projects/enpm690/Tactile_Guided_Insertion"
sys.path.insert(0, TACTILE_ROOT)

import env_setup  # registers FOTSPandaGripper
from env_setup.make_env import make_fots_env

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

# Tactile lowdim constants — must match preprocess_tactile_lowdim.py
POOL_H, POOL_W = 8, 8
BLOCK_H = 96 // POOL_H   # 12
BLOCK_W = 128 // POOL_W  # 16

# Keys the BC-RNN policy was trained on
POLICY_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object_pos",
    "object_quat",
    "object_to_robot0_eef_pos",
    "object_to_robot0_eef_quat",
    "tactile_left_lowdim",
    "tactile_right_lowdim",
]

# Robosuite native nut keys -> generic names used during training
KEY_REMAP = {
    "RoundNut_pos":                 "object_pos",
    "RoundNut_quat":                "object_quat",
    "RoundNut_to_robot0_eef_pos":   "object_to_robot0_eef_pos",
    "RoundNut_to_robot0_eef_quat":  "object_to_robot0_eef_quat",
    "SquareNut_pos":                "object_pos",
    "SquareNut_quat":               "object_quat",
    "SquareNut_to_robot0_eef_pos":  "object_to_robot0_eef_pos",
    "SquareNut_to_robot0_eef_quat": "object_to_robot0_eef_quat",
}


def tactile_to_lowdim(frame, baseline):
    """Convert one tactile frame (96x128x3 uint8) to 64-dim feature."""
    diff = frame.astype(np.float32) - baseline.astype(np.float32)
    gray = diff @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    pooled = gray.reshape(POOL_H, BLOCK_H, POOL_W, BLOCK_W).mean(axis=(1, 3))
    return (pooled / 128.0).ravel().astype(np.float32)


def process_obs(raw_obs, baseline_l, baseline_r):
    """Remap keys and inject lowdim tactile features for the policy."""
    obs = {KEY_REMAP.get(k, k): v for k, v in raw_obs.items()}
    obs["tactile_left_lowdim"]  = tactile_to_lowdim(raw_obs["tactile_left"],  baseline_l)
    obs["tactile_right_lowdim"] = tactile_to_lowdim(raw_obs["tactile_right"], baseline_r)
    return {k: obs[k] for k in POLICY_OBS_KEYS if k in obs}


def run_rollout(policy, env, horizon, video_writer=None, video_skip=5):
    policy.start_episode()
    raw_obs = env.reset()

    # First obs is contact-free -> use as per-episode tactile baseline
    baseline_l = raw_obs["tactile_left"].copy()
    baseline_r = raw_obs["tactile_right"].copy()
    obs = process_obs(raw_obs, baseline_l, baseline_r)

    total_reward = 0.0
    success = False
    video_count = 0

    for step in range(horizon):
        act = policy(ob=obs)
        raw_obs, reward, done, _ = env.step(act)

        total_reward += reward
        success = bool(env._check_success())
        obs = process_obs(raw_obs, baseline_l, baseline_r)

        if video_writer is not None and video_count % video_skip == 0:
            frame = raw_obs.get("agentview_image")
            if frame is not None:
                video_writer.append_data(frame[::-1])  # MuJoCo renders upside-down

        video_count += 1
        if done or success:
            break

    return {"Return": total_reward, "Horizon": step + 1, "Success": float(success)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",      required=True,       help="path to .pth checkpoint")
    parser.add_argument("--n_rollouts", type=int, default=10)
    parser.add_argument("--horizon",    type=int, default=400)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=args.agent, device=device, verbose=True
    )

    print("\n=== Creating FOTS environment ===")
    # _init_fots_engine resolves fots_sim/ relative to cwd, so run from TACTILE_ROOT
    _orig_cwd = os.getcwd()
    os.chdir(TACTILE_ROOT)
    try:
        env = make_fots_env(
            env_name="NutAssemblySingle",
            nut_type="round",
            fidelity_mode=True,
            render_height=96,
            render_width=128,
            has_offscreen_renderer=True,
            use_camera_obs=True,
        )
    finally:
        os.chdir(_orig_cwd)

    video_writer = None
    if args.video_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.video_path)), exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)

    all_stats = []
    for i in range(args.n_rollouts):
        print(f"\n--- Rollout {i + 1}/{args.n_rollouts} ---")
        stats = run_rollout(policy, env, args.horizon, video_writer, args.video_skip)
        all_stats.append(stats)
        print(json.dumps(stats, indent=2))

    if video_writer:
        video_writer.close()
        print(f"\nVideo saved to {args.video_path}")

    avg = {k: float(np.mean([s[k] for s in all_stats])) for k in all_stats[0]}
    avg["Num_Success"] = int(np.sum([s["Success"] for s in all_stats]))
    print("\n=== Average Rollout Stats ===")
    print(json.dumps(avg, indent=4))


if __name__ == "__main__":
    main()
