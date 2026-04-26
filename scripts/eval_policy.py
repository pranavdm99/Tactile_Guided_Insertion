#!/usr/bin/env python3
"""
Qualitative evaluation of a trained BC-RNN policy in simulation.

Loads a checkpoint, runs N rollouts through the FOTS NutAssembly environment,
saves a stitched video of each rollout (agentview | wrist | tactile_L | tactile_R),
and prints a final success-rate summary.

Usage (inside the docker container):
    python3 scripts/eval_policy.py --checkpoint checkpoints/best_policy.pth
    python3 scripts/eval_policy.py --checkpoint checkpoints/best_policy.pth --n_rollouts 10
    python3 scripts/eval_policy.py --checkpoint checkpoints/best_policy.pth --render
"""

import sys
import os
import argparse
import time

import numpy as np
import torch
import cv2

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env_setup.make_env import make_fots_env
from policy.bc_rnn import BCRNNPolicy


# ─────────────────────────────────────────────────────────────────────────── #
#  Obs remapping: env keys → policy keys                                      #
# ─────────────────────────────────────────────────────────────────────────── #


# Tactile training resolution — must match TactileEncoder(in_h=96, in_w=128)
_TAC_H, _TAC_W = 96, 128


def remap_obs(env_obs: dict, device: torch.device) -> dict:
    """
    Convert raw env observation dict to the (1, ...) tensor dict expected by
    BCRNNPolicy.act(). Mirrors the key renaming done in TactileInsertionDataset.

    env key                      → policy key
    ─────────────────────────────────────────
    agentview_image              → agentview_image
    robot0_eye_in_hand_image     → wrist_image
    tactile_left                 → tactile_left   (resized to 96×128)
    tactile_right                → tactile_right  (resized to 96×128)
    robot0_eef_pos               → eef_pos
    robot0_eef_quat              → eef_quat
    robot0_gripper_qpos          → gripper_qpos

    NOTE: The TactileEncoder's SpatialSoftmax grids are baked in for the
    training resolution (96×128). The env renders tactile at 320×240, so we
    resize here to avoid a shape mismatch.
    """
    def to_tensor(arr, dtype=torch.float32):
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32 if dtype == torch.float32 else np.uint8))
        return t.unsqueeze(0).to(device)   # (1, ...)

    def resize_tactile(img: np.ndarray) -> np.ndarray:
        """Resize tactile image (H, W, 3) uint8 to (_TAC_H, _TAC_W, 3)."""
        if img.shape[:2] == (_TAC_H, _TAC_W):
            return img
        return cv2.resize(img, (_TAC_W, _TAC_H), interpolation=cv2.INTER_LINEAR)

    return {
        "agentview_image": to_tensor(env_obs["agentview_image"],                         torch.uint8),
        "wrist_image":     to_tensor(env_obs["robot0_eye_in_hand_image"],                 torch.uint8),
        "tactile_left":    to_tensor(resize_tactile(env_obs["tactile_left"]),             torch.uint8),
        "tactile_right":   to_tensor(resize_tactile(env_obs["tactile_right"]),            torch.uint8),
        "eef_pos":         to_tensor(env_obs["robot0_eef_pos"]),
        "eef_quat":        to_tensor(env_obs["robot0_eef_quat"]),
        "gripper_qpos":    to_tensor(env_obs["robot0_gripper_qpos"]),
    }


# ─────────────────────────────────────────────────────────────────────────── #
#  Video helpers                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def make_frame(env_obs: dict, step: int, reward: float, done: bool) -> np.ndarray:
    """
    Build a single annotated side-by-side frame from the 4 image streams.

    Layout (all panels scaled to 256×256):
        [ agentview | wrist | tactile_L | tactile_R ]
    """
    TARGET = 256   # display size for every panel

    def prep(img: np.ndarray) -> np.ndarray:
        """uint8 (H,W,3) or float (H,W) → BGR uint8 256×256."""
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if img.ndim == 2:                          # grayscale depth
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img, (TARGET, TARGET), interpolation=cv2.INTER_LINEAR)

    panels = [
        prep(env_obs["agentview_image"]),
        prep(env_obs["robot0_eye_in_hand_image"]),
        prep(env_obs["tactile_left"]),
        prep(env_obs["tactile_right"]),
    ]

    frame = np.concatenate(panels, axis=1)   # (256, 1024, 3) BGR

    # ── Overlay text ──────────────────────────────────────────────── #
    labels = ["AgentView", "Wrist Cam", "Tactile L", "Tactile R"]
    for i, label in enumerate(labels):
        cv2.putText(frame, label, (i * TARGET + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (i * TARGET + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30),   1, cv2.LINE_AA)

    status_col = (0, 200, 0) if done else (200, 200, 200)
    cv2.putText(frame, f"Step {step:03d}  |  R={reward:.3f}",
                (6, TARGET - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 1, cv2.LINE_AA)

    return frame


def save_video(frames: list, path: str, fps: int = 20) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Qualitative BC-RNN policy evaluation")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_policy.pth",
                        help="Path to .pth checkpoint (relative to repo root or absolute)")
    parser.add_argument("--nut_type",   type=str, default=None,
                        choices=["round", "square"],
                        help="Nut type to evaluate on: 'round' or 'square'. "
                             "Omit to randomize per episode (default: random).")
    parser.add_argument("--n_rollouts",  type=int, default=5,
                        help="Number of rollout episodes to run (default: 5)")
    parser.add_argument("--horizon",     type=int, default=500,
                        help="Max steps per episode (default: 500)")
    parser.add_argument("--out_dir",     type=str, default="eval_videos",
                        help="Directory to write rollout videos (default: eval_videos/)")
    parser.add_argument("--fps",         type=int, default=20,
                        help="Video frame rate (default: 20)")
    parser.add_argument("--render",      action="store_true",
                        help="Show live MuJoCo viewer (requires display; off by default)")
    parser.add_argument("--no_video",    action="store_true",
                        help="Skip saving videos (faster dry-run)")
    parser.add_argument("--fidelity",    action="store_true", default=True,
                        help="Use FOTS fidelity (RGB) tactile mode (default: True)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="RNG seed (default: 0)")
    parser.add_argument("--gmm_modes",   type=int, default=3)
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────── #
    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
                else os.path.join(ROOT, args.checkpoint)
    if not os.path.exists(ckpt_path):
        print(f"❌  Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
              else os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────── #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  BC-RNN Policy — Qualitative Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")
    print(f"  Rollouts   : {args.n_rollouts}")
    print(f"  Nut type   : {args.nut_type if args.nut_type else 'random (per episode)'}")
    print(f"  Horizon    : {args.horizon} steps")
    print(f"  Output     : {out_dir}/")
    print(f"{'='*60}\n")

    # ── Load policy ──────────────────────────────────────────────────── #
    policy = BCRNNPolicy(num_gmm_modes=args.gmm_modes, freeze_visual=False, use_aux_loss=True).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
    policy.load_state_dict(ckpt[sd_key])
    policy.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", None)
    print(f"[CKPT] Loaded epoch {epoch}" + (f"  val_loss={val_loss:.4f}" if val_loss else ""))

    # ── Build environment ────────────────────────────────────────────── #
    print("[ENV]  Building FOTS NutAssembly environment …")
    env = make_fots_env(
        fidelity_mode=args.fidelity,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        nut_type=args.nut_type,         # None = random, 'round'/'square' = fixed
    )
    print("[ENV]  Ready.\n")

    # ── Rollout loop ─────────────────────────────────────────────────── #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = []   # list of (success: bool, total_reward: float, steps: int)

    for ep in range(args.n_rollouts):
        print(f"── Episode {ep + 1}/{args.n_rollouts} ──────────────────────────────")
        t0 = time.time()

        obs = env.reset()
        policy.reset()

        frames       = []
        total_reward = 0.0
        success      = False

        for step in range(args.horizon):
            # Convert obs and get action
            policy_obs = remap_obs(obs, device)
            action     = policy.act(policy_obs)   # (7,) numpy

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Check success via robosuite's internal metric
            if hasattr(env, "_check_success"):
                success = bool(env._check_success())
            elif "success" in info:
                success = bool(info["success"])

            # Collect frame
            if not args.no_video:
                frames.append(make_frame(obs, step, reward, success))

            if args.render:
                env.render()

            if success or done:
                break

        elapsed = time.time() - t0
        status  = "✅ SUCCESS" if success else "❌ FAILED "
        print(f"    {status}  |  steps={step+1:3d}  |  R={total_reward:.3f}  |  {elapsed:.1f}s")

        # Save video
        if not args.no_video and frames:
            video_path = os.path.join(out_dir, f"rollout_{ep+1:02d}_{'SUCCESS' if success else 'FAIL'}.mp4")
            save_video(frames, video_path, fps=args.fps)

        results.append((success, total_reward, step + 1))

    # ── Summary ──────────────────────────────────────────────────────── #
    n_success    = sum(r[0] for r in results)
    mean_reward  = np.mean([r[1] for r in results])
    mean_steps   = np.mean([r[2] for r in results])
    success_rate = n_success / args.n_rollouts * 100

    print(f"\n{'='*60}")
    print(f"  RESULTS — {args.n_rollouts} rollouts")
    print(f"{'='*60}")
    print(f"  Success rate  : {n_success}/{args.n_rollouts}  ({success_rate:.1f}%)")
    print(f"  Mean reward   : {mean_reward:.3f}")
    print(f"  Mean steps    : {mean_steps:.1f}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
