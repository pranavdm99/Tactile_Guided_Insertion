#!/usr/bin/env python3
"""
CLI training entry point for the simplified BC-RNN policy.

Usage examples:
    # Standard run with W&B logging
    python3 scripts/train_bc.py --data datasets/ --wandb-project tactile-bc

    # Quick smoke test (no wandb)
    python3 scripts/train_bc.py --data datasets/ --epochs 5 --batch-size 16 --no-wandb

Checkpoints are saved to <output>/best_policy.pth and <output>/latest_policy.pth.
Training config is written to <output>/config.json.
"""

from __future__ import annotations

import argparse
import glob
import h5py
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project imports ─────────────────────────────────────────────────────── #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from policy.bc_rnn import BCRNNPolicy
from policy.train  import TactileInsertionDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train simplified BC-RNN policy for tactile-guided insertion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data", type=str, required=True,
                   help="Path to an HDF5 file or directory of HDF5 files. "
                        "Raw fast-mode depth maps are used directly.")
    p.add_argument("--output", type=str, default="checkpoints",
                   help="Directory to save checkpoints and config.")
    # Sequence / data loading
    p.add_argument("--seq-len",     type=int,   default=20,
                   help="LSTM context window length (steps).")
    p.add_argument("--val-split",   type=float, default=0.2,
                   help="Fraction of demos held out for validation.")
    p.add_argument("--num-workers", type=int,   default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--cache-to-ram", action="store_true",
                   help="Load the entire dataset into RAM at the start.")
    # Training
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--clip-grad",    type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # W&B
    p.add_argument("--wandb-project", type=str, default="tactile-bc",
                   help="Weights & Biases project name.")
    p.add_argument("--wandb-run",     type=str, default=None,
                   help="W&B run name (auto-generated if not set).")
    p.add_argument("--no-wandb",      action="store_true",
                   help="Disable W&B logging.")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume training from.")
    return p.parse_args()


def encoder_grad_norms(policy) -> dict:
    modules = [
        ("enc_tac",     policy.enc_tac),
        ("enc_proprio", policy.enc_proprio),
        ("lstm",        policy.lstm),
        ("action_head", policy.action_head),
    ]
    norms = {}
    for name, mod in modules:
        params_with_grad = [p for p in mod.parameters() if p.grad is not None]
        if params_with_grad:
            total = torch.sqrt(sum(p.grad.norm() ** 2 for p in params_with_grad))
            norms[f"grad/{name}"] = total.item()
        else:
            norms[f"grad/{name}"] = 0.0
    return norms


def collect_files(target: str) -> list[str]:
    if os.path.isfile(target):
        return [target]
    elif os.path.isdir(target):
        files = sorted(
            glob.glob(os.path.join(target, "**/*.hdf5"), recursive=True) +
            glob.glob(os.path.join(target, "**/*.h5"),   recursive=True)
        )
        return files
    else:
        return []


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    print(f"[Train] Preparing dataset from: {args.data}")
    data_files = collect_files(args.data)
    if not data_files:
        print(f"[ERROR] No usable HDF5 files found in '{args.data}'.")
        sys.exit(1)

    print(f"[Train] {len(data_files)} file(s) found.")

    _all_demos: list[tuple[str, str]] = []
    for _p in data_files:
        with h5py.File(_p, "r") as _f:
            if "data" not in _f:
                continue
            for _dk in sorted(_f["data"].keys()):
                if "rewards" not in _f["data"][_dk]:
                    continue
                _r = _f["data"][_dk]["rewards"][:]
                if _r.max() < 0.5 or len(_r) < max(20, args.seq_len):
                    continue
                _all_demos.append((_p, _dk))

    _rng  = np.random.default_rng(args.seed)
    _perm = _rng.permutation(len(_all_demos)).tolist()
    _all_demos = [_all_demos[i] for i in _perm]
    _n_val = max(1, int(len(_all_demos) * args.val_split))
    val_demo_set   = set(_all_demos[-_n_val:])
    train_demo_set = set(_all_demos[:-_n_val])

    print(f"[Train] Demo split: {len(train_demo_set)} train demos | {_n_val} val demos "
          f"({len(_all_demos)} valid total)")

    train_ds = TactileInsertionDataset(
        data_paths=data_files,
        seq_len=args.seq_len,
        augment=False,
        cache_to_ram=args.cache_to_ram,
        allowed_demos=train_demo_set,
    )
    val_ds = TactileInsertionDataset(
        data_paths=data_files,
        seq_len=args.seq_len,
        augment=False,
        cache_to_ram=args.cache_to_ram,
        allowed_demos=val_demo_set,
        action_stats=(train_ds.action_mean, train_ds.action_std),
    )

    _pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=_pin,
    )

    action_mean = train_ds.action_mean
    action_std  = train_ds.action_std

    policy = BCRNNPolicy().to(device)

    print(
        f"[Model] {policy.num_parameters():,} total params | "
        f"{policy.num_trainable_parameters():,} trainable"
    )
    policy.set_action_norm(action_mean, action_std)

    _clip_head_params  = list(policy.action_head.parameters())
    _clip_head_ids     = {id(p) for p in _clip_head_params}
    _clip_other_params = [p for p in policy.parameters() if id(p) not in _clip_head_ids]

    optimizer = optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch    = 1
    best_val_loss  = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Resume] Epoch {start_epoch}, best val loss = {best_val_loss:.4f}")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
                resume="allow" if args.resume else None,
            )
            wandb.watch(policy, log="gradients", log_freq=100)
            print(f"[W&B] Logging to project '{args.wandb_project}'")
        except ImportError:
            print("[W&B] wandb not installed — run: pip install wandb")
            use_wandb = False
            
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    for epoch in range(start_epoch, args.epochs + 1):
        policy.train()
        train_losses, bc_losses, grad_norms = [], [], []
        encoder_norms_accum: dict[str, list] = {}
        global_step = (epoch - 1) * len(train_loader)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)):
            obs     = {k: v.to(device) for k, v in batch["obs"].items()}
            actions = batch["actions"].to(device)
            rtg     = batch["rtg_weights"].to(device)

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss, info = policy(obs, actions, rtg)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            enc_norms = encoder_grad_norms(policy)
            for k, v in enc_norms.items():
                encoder_norms_accum.setdefault(k, []).append(v)

            head_norm = torch.nn.utils.clip_grad_norm_(_clip_head_params,  args.clip_grad)
            enc_norm  = torch.nn.utils.clip_grad_norm_(_clip_other_params, 5.0)
            grad_norm = head_norm
            
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            bc_losses.append(info["bc_loss"])
            grad_norms.append(grad_norm.item())

            if use_wandb and (step + 1) % 50 == 0:
                wandb.log({
                    "batch/loss":           loss.item(),
                    "batch/bc_loss":        info["bc_loss"],
                    "batch/grad_norm_head": head_norm.item(),
                    "batch/grad_norm_enc":  enc_norm.item(),
                }, step=global_step + step)

        scheduler.step()

        policy.eval()
        val_losses_ep, val_bc_ep = [], []
        diag_accum: dict[str, list] = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False):
                obs     = {k: v.to(device) for k, v in batch["obs"].items()}
                actions = batch["actions"].to(device)
                rtg     = batch["rtg_weights"].to(device)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss, info = policy(obs, actions, rtg)
                    
                    fused = policy._encode(obs)
                    lstm_out, _ = policy.lstm(fused)
                    diag = policy.action_head.diagnostics(lstm_out, actions)
                    
                val_losses_ep.append(loss.item())
                val_bc_ep.append(info["bc_loss"])
                
                for k, v in diag.items():
                    diag_accum.setdefault(k, []).append(v)

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses_ep)
        val_bc     = np.mean(val_bc_ep)
        bc_loss    = np.mean(bc_losses)
        grad_norm  = np.mean(grad_norms)
        lr_main    = optimizer.param_groups[0]["lr"]

        avg_enc_norms = {k: np.mean(v) for k, v in encoder_norms_accum.items()}
        avg_diag = {k: np.mean(v) for k, v in diag_accum.items()}

        print(
            f"[Epoch {epoch:3d}/{args.epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"(bc={bc_loss:.4f}  "
            f"grad={grad_norm:.3f}  lr={lr_main:.2e})"
        )
        print(
            f"  Gripper acc={avg_diag.get('metrics/gripper_accuracy', 0):.1%}  "
            f"Pose MAE={avg_diag.get('diagnostics/pose_mae', 0):.4f}"
        )

        if use_wandb:
            log_dict = {
                "epoch":                  epoch,
                "loss/train":             train_loss,
                "loss/val":               val_loss,
                "loss/train_bc":          bc_loss,
                "loss/val_bc":            val_bc,
                "train/grad_norm":        grad_norm,
                "train/lr_main":          lr_main,
            }
            log_dict.update(avg_enc_norms)
            log_dict.update(avg_diag)
            epoch_step = epoch * len(train_loader)
            wandb.log(log_dict, step=epoch_step)

        ckpt = {
            "epoch":       epoch,
            "state_dict":  policy.state_dict(),
            "val_loss":    float(val_loss),
            "train_loss":  float(train_loss),
            "args":        vars(args),
            "action_mean": action_mean.tolist(),
            "action_std":  action_std.tolist(),
        }

        torch.save(ckpt, os.path.join(args.output, "latest_policy.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt_path = os.path.join(args.output, "best_policy.pth")
            torch.save(ckpt, best_ckpt_path)
            print(f"  ↳ ✅ New best checkpoint saved  (val={val_loss:.4f})")
            if use_wandb:
                artifact = wandb.Artifact(
                    name="best_policy",
                    type="model",
                    description=f"Best BC-RNN checkpoint (val={val_loss:.4f}, epoch={epoch})",
                    metadata={"val_loss": val_loss, "epoch": epoch},
                )
                artifact.add_file(best_ckpt_path)
                wandb.log_artifact(artifact)

    print(f"\n[Done] Best val loss: {best_val_loss:.4f}")
    print(f"       Checkpoint → {os.path.join(args.output, 'best_policy.pth')}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
