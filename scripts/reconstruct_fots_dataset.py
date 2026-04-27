#!/usr/bin/env python3
"""
Offline FOTS Dataset Reconstruction
=====================================
Reads a "Fast Mode" HDF5 dataset containing raw normalized depth maps for
tactile_left / tactile_right, and writes a new HDF5 file (with a `_fots` suffix)
where those streams are replaced with photorealistic FOTS RGB renders.

All other observations (agentview_image, robot0_eye_in_hand_image, proprio, etc.),
actions, rewards, and dones are copied verbatim.

Usage:
    # Reconstruct a single file
    python3 scripts/reconstruct_fots_dataset.py datasets/demo_1234567890.hdf5

    # Reconstruct all HDF5 files in a directory
    python3 scripts/reconstruct_fots_dataset.py datasets/

    # Overwrite existing _fots files (skip check)
    python3 scripts/reconstruct_fots_dataset.py datasets/ --overwrite
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import h5py
import torch
import cv2
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fots_sim.mlp_model import MLP
from fots_sim.utils.mlp_render import MLPRender


# ---------------------------------------------------------------------------
# FOTS Engine
# ---------------------------------------------------------------------------

def load_fots_engine(device: torch.device) -> MLPRender:
    """Load FOTS MLP and assemble the MLPRender object."""
    base_dir = "/app/fots_sim"
    if not os.path.exists(base_dir):
        base_dir = os.path.join(os.path.dirname(__file__), "..", "fots_sim")

    bg_img   = np.load(os.path.join(base_dir, "assets/digit_bg.npy"))
    bg_depth = np.load(os.path.join(base_dir, "utils/ini_depth_extent.npy"))
    bg_mlp   = np.load(os.path.join(base_dir, "utils/ini_bg_mlp.npy"))

    model = MLP().to(device)
    model_path = os.path.join(base_dir, "models/mlp_n2c_r.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return MLPRender(background_img=bg_img, bg_depth=bg_depth, bg_render=bg_mlp, model=model)


def render_fots_rgb(fots_render: MLPRender, depth: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """
    Render a single FOTS frame from a normalized depth map.

    Args:
        fots_render: Initialized MLPRender object.
        depth:      Normalized depth map, shape (H, W), dtype float32 in [0, 1].
        baseline:   Per-episode baseline depth (no-contact state), same shape.

    Returns:
        RGB image, shape (H, W, 3), dtype uint8.
    """
    fots_render.bg_depth = baseline
    fots_render._pre_scaled_bg = baseline * fots_render._scale
    return fots_render.generate(depth)  # Returns uint8 RGB


# ---------------------------------------------------------------------------
# Per-file reconstruction
# ---------------------------------------------------------------------------

TACTILE_KEYS = ("tactile_left", "tactile_right")


def reconstruct_file(src_path: str, dst_path: str, fots_render: MLPRender) -> bool:
    """
    Copy src_path → dst_path, replacing tactile depth streams with FOTS RGB.

    Returns True on success, False if the source is already RGB (uint8) and
    reconstruction is unnecessary.
    """
    with h5py.File(src_path, "r") as src:
        # Quick dtype check on the first available tactile stream
        data_grp = src["data"]
        first_demo = list(data_grp.keys())[0]
        sample_tactile = data_grp[f"{first_demo}/obs/tactile_left"]

        if sample_tactile.dtype == np.uint8:
            print(f"  [SKIP] {os.path.basename(src_path)} — tactile data is already uint8 RGB.")
            return False

        demo_keys = sorted(data_grp.keys(), key=lambda k: int(k.split("_")[1]))
        total_steps = sum(data_grp[f"{dk}/obs/tactile_left"].shape[0] for dk in demo_keys)

        print(f"  Processing {len(demo_keys)} demos ({total_steps} total steps)...")

        with h5py.File(dst_path, "w") as dst:
            # 1. Copy root-level attributes (env_name, total, etc.)
            for k, v in src.attrs.items():
                dst.attrs[k] = v

            dst_data = dst.create_group("data")

            # 2. Copy data group attributes (env_args, etc.)
            for k, v in data_grp.attrs.items():
                dst_data.attrs[k] = v

            # 3. Iterate demos
            with tqdm(total=total_steps, unit="step", leave=False) as pbar:
                for dk in demo_keys:
                    src_ep  = data_grp[dk]
                    dst_ep  = dst_data.create_group(dk)

                    # Copy episode-level attributes (num_samples, nut_type, etc.)
                    for k, v in src_ep.attrs.items():
                        dst_ep.attrs[k] = v

                    # Copy actions, rewards, dones verbatim
                    for key in ("actions", "rewards", "dones"):
                        dst_ep.create_dataset(key, data=src_ep[key][:], compression="gzip")

                    # Process observations
                    src_obs = src_ep["obs"]
                    dst_obs = dst_ep.create_group("obs")

                    # 3a. Load raw tactile depth for this demo
                    depth_l_all = src_obs["tactile_left"][:]   # (T, H, W) float32
                    depth_r_all = src_obs["tactile_right"][:]  # (T, H, W) float32

                    # 3b. Per-episode baselines = first frame (no-contact state)
                    baseline_l = depth_l_all[0]
                    baseline_r = depth_r_all[0]

                    T = depth_l_all.shape[0]
                    H, W = depth_l_all.shape[1], depth_l_all.shape[2]

                    fots_l_frames = np.empty((T, H, W, 3), dtype=np.uint8)
                    fots_r_frames = np.empty((T, H, W, 3), dtype=np.uint8)

                    for t in range(T):
                        fots_l_frames[t] = render_fots_rgb(fots_render, depth_l_all[t], baseline_l)
                        fots_r_frames[t] = render_fots_rgb(fots_render, depth_r_all[t], baseline_r)
                        pbar.update(1)

                    dst_obs.create_dataset("tactile_left",  data=fots_l_frames, compression="gzip")
                    dst_obs.create_dataset("tactile_right", data=fots_r_frames, compression="gzip")

                    # 3c. Copy all other obs verbatim
                    for k in src_obs.keys():
                        if k in TACTILE_KEYS:
                            continue
                        dst_obs.create_dataset(k, data=src_obs[k][:], compression="gzip")

    return True


# ---------------------------------------------------------------------------
# Public API — importable by train_bc.py
# ---------------------------------------------------------------------------

def prepare_training_files(
    target: str,
    device: "torch.device | None" = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Given a file path or directory, ensure all HDF5 datasets have uint8 tactile
    data (FOTS RGB renders) and return a list of paths that are ready for training.

    Rules:
      - Already uint8  (Fidelity mode)  → original path returned as-is.
      - float32 depth  (Fast mode)      → reconstructed to ``_fots.hdf5``;
                                           the reconstructed path is returned.
      - FOTS engine is loaded only when at least one file needs reconstruction.

    Args:
        target:    Path to a single .hdf5 file or a directory of .hdf5 files.
        device:    Torch device for FOTS MLP inference (auto-detected if None).
        overwrite: If True, redo reconstruction even when a _fots file exists.

    Returns:
        List of absolute file paths ready for training (may be empty on error).
    """
    files = collect_files(target)
    if not files:
        return []

    ready:                list[str] = []
    needs_reconstruction: list[str] = []

    # Quick dtype probe
    for f in files:
        try:
            with h5py.File(f, "r") as hf:
                first_demo = sorted(hf["data"].keys(), key=lambda k: int(k.split("_")[1]))[0]
                dtype = hf[f"data/{first_demo}/obs/tactile_left"].dtype
            if dtype == np.uint8:
                print(f"  [READY] {os.path.basename(f)} — already uint8 RGB, no reconstruction needed.")
                ready.append(f)
            else:
                needs_reconstruction.append(f)
        except Exception as e:
            print(f"  [WARN] Could not probe {os.path.basename(f)}: {e} — skipping.")

    # Reconstruct only files that need it
    if needs_reconstruction:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading FOTS engine on {device} for {len(needs_reconstruction)} file(s)...")
        fots_render = load_fots_engine(device)
        print("[INFO] FOTS engine ready.")

        for src_path in needs_reconstruction:
            dst_path = make_dst_path(src_path)
            if os.path.exists(dst_path) and not overwrite:
                print(f"  [CACHED] {os.path.basename(dst_path)} already exists, using it.")
                ready.append(dst_path)
                continue
            try:
                success = reconstruct_file(src_path, dst_path, fots_render)
                ready.append(dst_path if success else src_path)
            except Exception as e:
                print(f"  [ERROR] Reconstruction failed for {os.path.basename(src_path)}: {e}")

    return ready


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_files(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    elif os.path.isdir(target):
        files = sorted(
            glob.glob(os.path.join(target, "**/*.hdf5"), recursive=True) +
            glob.glob(os.path.join(target, "**/*.h5"),   recursive=True)
        )
        # Include all HDF5 files; ready-to-train files will be identified by dtype later
        return files
    else:
        print(f"[ERROR] Path does not exist: {target}")
        return []


def make_dst_path(src_path: str) -> str:
    base, ext = os.path.splitext(src_path)
    return f"{base}_fots{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Offline FOTS reconstruction: convert raw-depth HDF5 → FOTS RGB HDF5."
    )
    parser.add_argument("target", type=str,
                        help="Path to a single HDF5 file or a directory of HDF5 files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing _fots files (default: skip).")
    args = parser.parse_args()

    files = collect_files(args.target)

    if not files:
        print("[ERROR] No source files found.")
        sys.exit(1)

    print(f"[INFO] Found {len(files)} file(s) to process.")

    # Load FOTS engine once for all files
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading FOTS engine on {device}...")
    fots_render = load_fots_engine(device)
    print("[INFO] FOTS engine ready.\n")

    ok, skipped, failed = 0, 0, 0

    for src_path in files:
        dst_path = make_dst_path(src_path)
        fname    = os.path.basename(src_path)

        print(f"[{ok + skipped + failed + 1}/{len(files)}] {fname}")

        if os.path.exists(dst_path) and not args.overwrite:
            print(f"  [SKIP] Output already exists: {os.path.basename(dst_path)}")
            skipped += 1
            continue

        try:
            success = reconstruct_file(src_path, dst_path, fots_render)
            if success:
                # Sanity check output
                with h5py.File(dst_path, "r") as f:
                    first = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[0]
                    dtype = f[f"data/{first}/obs/tactile_left"].dtype
                    assert dtype == np.uint8, f"Expected uint8, got {dtype}"
                print(f"  [OK] → {os.path.basename(dst_path)}")
                ok += 1
            else:
                skipped += 1
                # Clean up empty dst file if it was created
                if os.path.exists(dst_path) and os.path.getsize(dst_path) == 0:
                    os.remove(dst_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            if os.path.exists(dst_path):
                os.remove(dst_path)
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Done. ✅ {ok} reconstructed | ⏭️  {skipped} skipped | ❌ {failed} failed")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
