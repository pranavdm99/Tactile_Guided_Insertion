"""
Dataset, Return-to-Go computation, and augmentation utilities for BC-RNN training.

TactileInsertionDataset:
    - Reads FOTS-reconstructed *_fots.hdf5 files.
    - Filters failed demos (max reward == 0).
    - Computes per-step Return-to-Go weights.
    - Samples contiguous windows of length seq_len.
    - Applies visual augmentation (color jitter) consistently across the window.
"""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────── #
#  Return-to-Go computation                                                   #
# ─────────────────────────────────────────────────────────────────────────── #

def compute_rtg(
    rewards: np.ndarray,
    gamma:   float = 0.99,
    floor:   float = 0.1,
) -> np.ndarray:
    """
    Compute normalised, discounted Return-to-Go weights for one trajectory.

    R_t = Σ_{k≥t} γ^(k-t) · r_k,  then normalised to [0, 1] and clipped to [floor, 1].

    Using a non-zero floor ensures that early reaching steps contribute to
    training even when the shaped reward is small at the start of the episode.

    Args:
        rewards: (T,)  per-step reward array (dense Robosuite staged rewards).
        gamma:         Discount factor (default 0.99).
        floor:         Minimum weight — prevents early steps from being zeroed out.

    Returns:
        (T,) float32 weights in [floor, 1.0].
    """
    T   = len(rewards)
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running
        rtg[t]  = running

    max_val = rtg.max()
    if max_val > 0.0:
        rtg = rtg / max_val

    return np.clip(rtg, floor, 1.0)


# ─────────────────────────────────────────────────────────────────────────── #
#  Dataset                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

# HDF5 keys expected in each demo's "obs" group
_IMG_KEYS = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "tactile_left",
    "tactile_right",
]
_FLOAT_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object_to_robot0_eef_pos",   # Used for auxiliary observer loss (optional)
]


class TactileInsertionDataset(Dataset):
    """
    Windowed HDF5 dataset for BC-RNN sequence training.

    The dataset indexes all contiguous windows of length `seq_len` across all
    successful demos (max reward > 0) from one or more *_fots.hdf5 files.

    Each __getitem__ returns:
        obs:         dict of (T, ...) tensors — see keys below.
        actions:     (T, 7) float32 ground-truth 7-DOF actions.
        rtg_weights: (T,)   float32 Return-to-Go weights ∈ [floor, 1.0].

    Obs dict keys:
        agentview_image     (T, H, W, 3) uint8
        wrist_image         (T, H, W, 3) uint8
        tactile_left        (T, H, W, 3) uint8
        tactile_right       (T, H, W, 3) uint8
        eef_pos             (T, 3)       float32
        eef_quat            (T, 4)       float32
        gripper_qpos        (T, 2)       float32
        obj_to_eef (opt)    (T, 3)       float32  — present when available

    Args:
        data_paths: List of absolute paths to *_fots.hdf5 files.
        seq_len:    Context window length T (default 20 = 1 s at 20 Hz).
        gamma:      RTG discount factor.
        floor:      RTG minimum weight.
        augment:    Apply visual augmentation (disable for validation).
    """

    def __init__(
        self,
        data_paths:    list[str],
        seq_len:       int   = 20,
        gamma:         float = 0.99,
        floor:         float = 0.1,
        augment:       bool  = True,
        cache_to_ram:  bool  = True,
        allowed_demos: set[tuple[str, str]] | None = None,
    ):
        self.seq_len       = seq_len
        self.gamma         = gamma
        self.floor         = floor
        self.augment       = augment
        self.cache_to_ram  = cache_to_ram
        self.allowed_demos = allowed_demos

        # List of (file_path, demo_key, start_step_index) tuples
        self._windows: list[tuple[str, str, int]] = []
        
        # Cache for HDF5 file handles (populated lazily per-worker)
        self._file_handles: dict[str, h5py.File] = {}
        
        # RAM Cache
        self._data_cache: dict[str, dict[str, dict[str, np.ndarray]]] = {}

        for path in data_paths:
            self._index_file(path)

        if self.cache_to_ram:
            print(f"[Dataset] Pre-loading {len(data_paths)} files into RAM...")
            for path in data_paths:
                self._load_file_to_ram(path)

        if len(self._windows) == 0:
            raise RuntimeError(
                f"No successful demos (max_reward > 0) with ≥ {seq_len} steps "
                f"found in: {data_paths}\n"
                "→ Make sure you have run reconstruct_fots_dataset.py and that "
                "your demos were recorded after enabling reward_shaping=True."
            )

        print(
            f"[Dataset] {len(self._windows):,} windows "
            f"| seq_len={seq_len} | augment={augment} | cache_to_ram={cache_to_ram}"
        )

    def _load_file_to_ram(self, path: str) -> None:
        self._data_cache[path] = {}
        with h5py.File(path, "r") as f:
            data = f["data"]
            for dk in data.keys():
                if self.allowed_demos is not None and (path, dk) not in self.allowed_demos:
                    continue
                demo = data[dk]
                self._data_cache[path][dk] = {
                    "obs": {k: demo["obs"][k][:] for k in demo["obs"].keys()},
                    "actions": demo["actions"][:],
                    "rewards": demo["rewards"][:],
                }

    # ──────────────────────────────────────────────── #

    def _index_file(self, path: str) -> None:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return
            data = f["data"]
            for dk in data.keys():
                if self.allowed_demos is not None and (path, dk) not in self.allowed_demos:
                    continue
                if "rewards" not in data[dk]:
                    continue
                rewards  = data[dk]["rewards"][:]
                T_demo   = len(rewards)
                if rewards.max() < 0.5 or T_demo < max(100, self.seq_len):
                    continue
                for start in range(T_demo - self.seq_len + 1):
                    self._windows.append((path, dk, start))

    # ──────────────────────────────────────────────── #

    def __len__(self) -> int:
        return len(self._windows)

    # ──────────────────────────────────────────────── #

    def __getitem__(self, idx: int) -> dict:
        path, dk, start = self._windows[idx]
        end = start + self.seq_len

        if self.cache_to_ram:
            demo    = self._data_cache[path][dk]
            obs_raw = demo["obs"]
            
            # ── Observations ──
            obs = {}
            for k in _IMG_KEYS:
                if k in obs_raw:
                    obs[k] = obs_raw[k][start:end]
                else:
                    ref = obs_raw["agentview_image"]
                    obs[k] = np.zeros((self.seq_len, *ref.shape[1:]), dtype=np.uint8)
            
            for k in _FLOAT_KEYS:
                if k in obs_raw:
                    obs[k] = obs_raw[k][start:end].astype(np.float32)
            
            # ── Actions and RTG ──
            actions = demo["actions"][start:end].astype(np.float32)
            rewards = demo["rewards"][:].astype(np.float32)
        else:
            # Lazy init HDF5 handle (keeps it open per-worker, removes massive I/O overhead)
            if path not in self._file_handles:
                self._file_handles[path] = h5py.File(path, "r", swmr=True)
                
            f = self._file_handles[path]
            demo    = f[f"data/{dk}"]
            obs_raw = demo["obs"]

            # ── Image observations ──────────────────────────────────── #
            obs = {}
            for k in _IMG_KEYS:
                if k in obs_raw:
                    obs[k] = obs_raw[k][start:end]    # (T, H, W, 3) uint8
                else:
                    # Black fallback — e.g. pre-wrist-cam datasets
                    ref   = obs_raw["agentview_image"]
                    obs[k] = np.zeros((self.seq_len, *ref.shape[1:]), dtype=np.uint8)

            # ── Float observations ──────────────────────────────────── #
            for k in _FLOAT_KEYS:
                if k in obs_raw:
                    obs[k] = obs_raw[k][start:end].astype(np.float32)

            # ── Actions and RTG ──────────────────────────────────────── #
            actions = demo["actions"][start:end].astype(np.float32)     # (T, 7)
            rewards = demo["rewards"][:].astype(np.float32)             # full demo

        # ── Actions and RTG ──
        rtg_all = compute_rtg(rewards, gamma=self.gamma, floor=self.floor)
        rtg_seq = rtg_all[start:end]                                # (T,)

        # ── Visual augmentation (MOVING TO GPU FOR SPEED) ───────────── #
        # if self.augment:
        #     obs = self._augment_visual(obs)

        # ── Build item dict ───────────────────────────────────────────── #
        item = {
            "obs": {
                "agentview_image": torch.from_numpy(obs["agentview_image"]),
                "wrist_image":     torch.from_numpy(obs["robot0_eye_in_hand_image"]),
                "tactile_left":    torch.from_numpy(obs["tactile_left"]),
                "tactile_right":   torch.from_numpy(obs["tactile_right"]),
                "eef_pos":         torch.from_numpy(obs["robot0_eef_pos"]),
                "eef_quat":        torch.from_numpy(obs["robot0_eef_quat"]),
                "gripper_qpos":    torch.from_numpy(obs["robot0_gripper_qpos"]),
            },
            "actions":     torch.from_numpy(actions),
            "rtg_weights": torch.from_numpy(rtg_seq),
        }

        # Auxiliary observer target — present in newer datasets only
        if "object_to_robot0_eef_pos" in obs:
            item["obs"]["obj_to_eef"] = torch.from_numpy(
                obs["object_to_robot0_eef_pos"]
            )

        return item

    # ──────────────────────────────────────────────── #
    #  Augmentation                                    #
    # ──────────────────────────────────────────────── #

    @staticmethod
    def _augment_visual(obs: dict) -> dict:
        """
        Apply consistent random colour jitter to all *camera* frames in the
        window (same random parameters for every t to preserve temporal coherence).

        Tactile images are NOT augmented — their colour statistics carry
        physical meaning (gel deformation → colour mapping by FOTS MLP).
        """
        brightness = random.uniform(0.85, 1.15)
        contrast   = random.uniform(0.85, 1.15)
        saturation = random.uniform(0.85, 1.15)
        hue        = random.uniform(-0.05, 0.05)

        for key in ("agentview_image", "robot0_eye_in_hand_image"):
            imgs = obs[key]                                      # (T, H, W, 3) np.uint8
            
            # Vectorised augmentation over T dimension
            img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)   # (T, 3, H, W)
            
            img_t = TF.adjust_brightness(img_t, brightness)
            img_t = TF.adjust_contrast(img_t, contrast)
            img_t = TF.adjust_saturation(img_t, saturation)
            img_t = TF.adjust_hue(img_t, hue)
            
            obs[key] = img_t.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

        return obs
