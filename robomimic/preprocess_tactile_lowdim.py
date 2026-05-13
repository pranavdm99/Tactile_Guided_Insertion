"""
Convert FOTS tactile images → 64-dim low-dim features and merge all demo HDF5 files
into one clean dataset suitable for fast low-dim BC-RNN training.

Tactile feature extraction per sensor per timestep:
  1. Compute per-episode baseline = mean of first 5 frames (robot not touching)
  2. Subtract baseline, convert to grayscale (0.299R + 0.587G + 0.114B)
  3. Average-pool to 8x8 grid  ->  64 values
  4. Normalize by 128.0  ->  roughly [-1, 1]

Output dataset contains only low-dim keys (no images), making it fast to cache.
"""

import os
import h5py
import numpy as np
from tqdm import tqdm

DATASETS_DIR = "/media/tirth/Expansion/docker_data_mount/projects/enpm690/Tactile_Guided_Insertion/datasets"
OUTPUT_PATH  = "/media/tirth/Expansion/docker_data_mount/projects/enpm690/Tactile_Guided_Insertion/datasets/merged_lowdim.hdf5"

FOTS_FILES = sorted([
    os.path.join(DATASETS_DIR, f)
    for f in os.listdir(DATASETS_DIR)
    if f.endswith("_fots.hdf5")
])

LOW_DIM_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object_pos",
    "object_quat",
    "object_to_robot0_eef_pos",
    "object_to_robot0_eef_quat",
]

BASELINE_FRAMES = 5   # frames at episode start assumed to be contact-free
POOL_H, POOL_W  = 8, 8
BLOCK_H = 96  // POOL_H   # 12
BLOCK_W = 128 // POOL_W   # 16


def tactile_to_lowdim(frames: np.ndarray) -> np.ndarray:
    """
    frames: (T, 96, 128, 3) uint8
    returns: (T, 64) float32, baseline-subtracted, pooled, normalised
    """
    T = frames.shape[0]
    baseline = frames[:BASELINE_FRAMES].mean(axis=0).astype(np.float32)   # (96,128,3)

    out = np.empty((T, POOL_H * POOL_W), dtype=np.float32)
    for t in range(T):
        diff  = frames[t].astype(np.float32) - baseline                   # (96,128,3)
        gray  = diff @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  # (96,128)
        pooled = gray.reshape(POOL_H, BLOCK_H, POOL_W, BLOCK_W).mean(axis=(1, 3))  # (8,8)
        out[t] = (pooled / 128.0).ravel()
    return out


def process_file(src_path: str, dst: h5py.File, ep_offset: int) -> int:
    src = h5py.File(src_path, "r")
    episodes = sorted(src["data"].keys())

    for ep_name in tqdm(episodes, desc=os.path.basename(src_path), leave=False):
        src_ep   = src["data"][ep_name]
        new_name = f"demo_{ep_offset}"
        ep_offset += 1

        dst_ep = dst["data"].require_group(new_name)

        # ── non-obs keys ────────────────────────────────────────────── #
        for key in ("actions", "rewards", "dones"):
            if key in src_ep:
                dst_ep.create_dataset(key, data=src_ep[key][:], compression="gzip")

        # ── low-dim obs ──────────────────────────────────────────────── #
        obs_grp = dst_ep.require_group("obs")
        src_obs  = src_ep["obs"]

        for key in LOW_DIM_OBS_KEYS:
            if key in src_obs:
                obs_grp.create_dataset(key, data=src_obs[key][:], compression="gzip")

        # ── tactile low-dim ──────────────────────────────────────────── #
        for tac_key, out_key in [("tactile_left",  "tactile_left_lowdim"),
                                  ("tactile_right", "tactile_right_lowdim")]:
            if tac_key in src_obs:
                ld = tactile_to_lowdim(src_obs[tac_key][:])
                obs_grp.create_dataset(out_key, data=ld, compression="gzip")

        # ── episode metadata ─────────────────────────────────────────── #
        dst_ep.attrs["num_samples"] = src_ep.attrs.get("num_samples",
                                         src_ep["actions"].shape[0])

    src.close()
    return ep_offset


def main():
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    dst = h5py.File(OUTPUT_PATH, "w")
    data_grp = dst.require_group("data")

    # Copy env_args from largest file so robomimic can reconstruct the env
    ref_file = max(FOTS_FILES, key=os.path.getsize)
    with h5py.File(ref_file, "r") as ref:
        for attr_key, attr_val in ref["data"].attrs.items():
            data_grp.attrs[attr_key] = attr_val

    ep_offset = 1
    for fpath in FOTS_FILES:
        ep_offset = process_file(fpath, dst, ep_offset)

    total = ep_offset - 1
    data_grp.attrs["total"] = total
    dst.close()

    print(f"\nDone. Merged {total} episodes → {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
