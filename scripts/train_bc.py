#!/usr/bin/env python3
"""
CLI training entry point for the BC-RNN policy.

Prerequisites:
    1. Collect demos:           python3 scripts/teleop_keyboard_mouse.py --nut round
    2. Reconstruct FOTS:        python3 scripts/reconstruct_fots_dataset.py datasets/
    3. Train:                   python3 scripts/train_bc.py --data datasets/

Usage examples:
    # Standard run with W&B logging
    python3 scripts/train_bc.py --data datasets/ --wandb-project tactile-bc

    # Quick smoke test (no wandb)
    python3 scripts/train_bc.py --data datasets/ --epochs 5 --batch-size 8 --no-wandb

    # No auxiliary loss, custom output dir
    python3 scripts/train_bc.py --data datasets/ --no-aux-loss --output runs/bc_no_aux

Checkpoints are saved to <output>/best_policy.pth and <output>/latest_policy.pth.
Training config is written to <output>/config.json.
W&B run logs are synced to the project specified by --wandb-project.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

# ── Project imports ─────────────────────────────────────────────────────── #
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # so reconstruct_fots_dataset is importable

from policy.bc_rnn import BCRNNPolicy
from policy.train  import TactileInsertionDataset
from reconstruct_fots_dataset import prepare_training_files


# ─────────────────────────────────────────────────────────────────────────── #
#  Argument parsing                                                           #
# ─────────────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BC-RNN policy for tactile-guided insertion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--data", type=str, required=True,
                   help="Path to an HDF5 file or directory of HDF5 files. "
                        "Fidelity-mode (uint8) files are used directly; "
                        "Fast-mode (float32) files are reconstructed via FOTS automatically.")
    p.add_argument("--output", type=str, default="checkpoints",
                   help="Directory to save checkpoints and config.")
    p.add_argument("--overwrite-fots", action="store_true",
                   help="Force re-reconstruction of Fast-mode files even if _fots.hdf5 exists.")
    # Sequence / data loading
    p.add_argument("--seq-len",     type=int,   default=20,
                   help="LSTM context window length (steps).")
    p.add_argument("--val-split",   type=float, default=0.1,
                   help="Fraction of windows held out for validation.")
    p.add_argument("--num-workers", type=int,   default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--cache-to-ram", action="store_true",
                   help="Load the entire dataset into RAM at the start (highly recommended for < 10GB datasets).")
    # Training
    p.add_argument("--epochs",       type=int,   default=200)
    p.add_argument("--batch-size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="Learning rate for non-visual parameters.")
    p.add_argument("--visual-lr",    type=float, default=1e-5,
                   help="LR for ResNet backbones after unfreeze.")
    p.add_argument("--unfreeze-epoch", type=int, default=5,
                   help="Epoch at which to unfreeze visual backbones.")
    p.add_argument("--clip-grad",    type=float, default=1.0,
                   help="Gradient clipping max norm.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # Architecture
    p.add_argument("--gmm-modes",   type=int,  default=3,
                   help="Number of GMM mixture components K.")
    p.add_argument("--no-aux-loss", action="store_true",
                   help="Disable the auxiliary observer loss.")
    # W&B
    p.add_argument("--wandb-project", type=str, default="tactile-bc",
                   help="Weights & Biases project name.")
    p.add_argument("--wandb-run",     type=str, default=None,
                   help="W&B run name (auto-generated if not set).")
    p.add_argument("--no-wandb",      action="store_true",
                   help="Disable W&B logging (useful for quick tests).")
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume training from.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                    #
# ─────────────────────────────────────────────────────────────────────────── #

def encoder_grad_norms(policy) -> dict:
    """
    Compute per-encoder gradient L2 norms after a backward pass.
    Useful for spotting stagnant encoders (norm ≈ 0) or unstable ones (norm >> 1).
    """
    modules = [
        ("enc_agent",   policy.enc_agent),
        ("enc_wrist",   policy.enc_wrist),
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


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                       #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    args = parse_args()

    # ── Reproducibility ─────────────────────────────────────────────── #
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # ── Resolve training-ready dataset files ─────────────────────────── #
    # prepare_training_files() inspects each file's tactile dtype:
    #   • uint8  (Fidelity mode) → returned as-is, no FOTS engine needed.
    #   • float32 (Fast mode)   → reconstructed to _fots.hdf5, that path returned.
    print(f"[Train] Preparing dataset from: {args.data}")
    data_files = prepare_training_files(
        args.data,
        device=device,
        overwrite=getattr(args, "overwrite_fots", False),
    )
    if not data_files:
        print(
            f"[ERROR] No usable HDF5 files found in '{args.data}'.\n"
            "Make sure the path points to a valid .hdf5 file or directory."
        )
        sys.exit(1)

    print(f"[Train] {len(data_files)} training-ready file(s):")
    for f in data_files:
        print(f"  • {f}")


    # ── Dataset / DataLoader ─────────────────────────────────────────── #
    full_ds = TactileInsertionDataset(
        data_paths=data_files,
        seq_len=args.seq_len,
        augment=True,
        cache_to_ram=args.cache_to_ram,
    )

    val_size   = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Disable augmentation for validation subset
    # (val_ds wraps the same dataset object; use a simple flag approach)
    class _NoAugSubset(Subset):
        def __getitem__(self, idx):
            old = self.dataset.augment
            self.dataset.augment = False
            item = super().__getitem__(idx)
            self.dataset.augment = old
            return item
        def __getitems__(self, indices):
            return [self.__getitem__(idx) for idx in indices]

    val_ds = _NoAugSubset(full_ds, val_ds.indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"[Train] {train_size:,} train windows | {val_size:,} val windows")

    # ── Model ────────────────────────────────────────────────────────── #
    policy = BCRNNPolicy(
        num_gmm_modes=args.gmm_modes,
        freeze_visual=True,
        use_aux_loss=not args.no_aux_loss,
    ).to(device)

    print(
        f"[Model] {policy.num_parameters():,} total params | "
        f"{policy.num_trainable_parameters():,} trainable"
    )

    # ── Optimizer  ───────────────────────────────────────────────────── #
    # Visual backbone parameters are in a separate group so we can adjust
    # their LR independently after unfreeze.
    visual_params = (
        list(policy.enc_agent.parameters())
        + list(policy.enc_wrist.parameters())
    )
    visual_param_ids = {id(p) for p in visual_params}
    other_params = [p for p in policy.parameters() if id(p) not in visual_param_ids]

    optimizer = optim.AdamW(
        [
            {"params": other_params,  "lr": args.lr,        "name": "main"},
            {"params": visual_params, "lr": 0.0,            "name": "visual"},  # frozen
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Resume ───────────────────────────────────────────────────────── #
    start_epoch    = 1
    best_val_loss  = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"[Resume] Epoch {start_epoch}, best val loss = {best_val_loss:.4f}")

    # ── Output directory ─────────────────────────────────────────────── #
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── W&B initialisation ───────────────────────────────────────────── #
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
            # Track the model graph (logged once at first forward pass)
            wandb.watch(policy, log="gradients", log_freq=100)
            print(f"[W&B] Logging to project '{args.wandb_project}'")
        except ImportError:
            print("[W&B] wandb not installed — run: pip install wandb")
            use_wandb = False
            
    # Mixed precision scaler for 2x faster ResNet CNN backprop on modern GPUs
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # ── Training loop ────────────────────────────────────────────────── #
    for epoch in range(start_epoch, args.epochs + 1):

        # Unfreeze visual backbones after warm-up
        if epoch == args.unfreeze_epoch:
            policy.unfreeze_visual()
            for g in optimizer.param_groups:
                if g.get("name") == "visual":
                    g["lr"] = args.visual_lr
            print(f"\n[Epoch {epoch}] ▶ Visual backbones unfrozen (lr={args.visual_lr})\n")

        # ── Train ─────────────────────────────────────────────────── #
        policy.train()
        train_losses, bc_losses, aux_losses, grad_norms = [], [], [], []
        encoder_norms_accum: dict[str, list] = {}
        rtg_means, rtg_mins, rtg_maxs = [], [], []
        global_step = (epoch - 1) * len(train_loader)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)):
            obs     = {k: v.to(device) for k, v in batch["obs"].items()}
            actions = batch["actions"].to(device)
            rtg     = batch["rtg_weights"].to(device)

            # RTG stats (before forward pass)
            rtg_means.append(rtg.mean().item())
            rtg_mins.append(rtg.min().item())
            rtg_maxs.append(rtg.max().item())

            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                loss, info = policy(obs, actions, rtg)
            
            scaler.scale(loss).backward()

            # Global grad norm (requires unscaling for accurate clipping/logging)
            scaler.unscale_(optimizer)
            
            # Per-encoder grad norms (before clipping, after unscale)
            enc_norms = encoder_grad_norms(policy)
            for k, v in enc_norms.items():
                encoder_norms_accum.setdefault(k, []).append(v)
                
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip_grad)
            
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            bc_losses.append(info["bc_loss"])
            aux_losses.append(info["aux_loss"])
            grad_norms.append(grad_norm.item())

            # ── Intra-epoch batch logging (every 50 steps) ──────── #
            if use_wandb and (step + 1) % 50 == 0:
                wandb.log({
                    "batch/loss":      loss.item(),
                    "batch/bc_loss":   info["bc_loss"],
                    "batch/aux_loss":  info["aux_loss"],
                    "batch/grad_norm": grad_norm.item(),
                }, step=global_step + step)

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────── #
        policy.eval()
        val_losses_ep, val_bc_ep, val_aux_ep = [], [], []
        diag_accum: dict[str, list] = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False):
                obs     = {k: v.to(device) for k, v in batch["obs"].items()}
                actions = batch["actions"].to(device)
                rtg     = batch["rtg_weights"].to(device)

                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    loss, info = policy(obs, actions, rtg)
                    
                    # Run GMM diagnostics (reuse LSTM output via internal encode)
                    fused, _ = policy._encode(obs)
                    lstm_out, _ = policy.lstm(fused)
                    diag = policy.action_head.diagnostics(lstm_out, actions)
                    
                val_losses_ep.append(loss.item())
                val_bc_ep.append(info["bc_loss"])
                val_aux_ep.append(info["aux_loss"])
                
                for k, v in diag.items():
                    diag_accum.setdefault(k, []).append(v)

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses_ep)
        val_bc     = np.mean(val_bc_ep)
        val_aux    = np.mean(val_aux_ep)
        bc_loss    = np.mean(bc_losses)
        aux_loss   = np.mean(aux_losses)
        grad_norm  = np.mean(grad_norms)
        lr_main    = optimizer.param_groups[0]["lr"]
        lr_visual  = optimizer.param_groups[1]["lr"]

        # Averaged per-encoder grad norms
        avg_enc_norms = {k: np.mean(v) for k, v in encoder_norms_accum.items()}
        # Averaged GMM diagnostics over val batches
        avg_diag = {k: np.mean(v) for k, v in diag_accum.items()}

        print(
            f"[Epoch {epoch:3d}/{args.epochs}]  "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"(bc={bc_loss:.4f}  aux={aux_loss:.4f}  "
            f"grad={grad_norm:.3f}  lr={lr_main:.2e})"
        )
        print(
            f"  Gripper acc={avg_diag.get('metrics/gripper_accuracy', 0):.1%}  "
            f"Pose MAE={avg_diag.get('diagnostics/pose_mae', 0):.4f}  "
            f"GMM std={avg_diag.get('diagnostics/gmm_mean_std', 0):.4f}  "
            f"Entropy={avg_diag.get('diagnostics/gmm_weight_entropy', 0):.4f}"
        )

        # ── W&B logging ───────────────────────────────────────────── #
        if use_wandb:
            log_dict = {
                "epoch":                  epoch,
                # Losses
                "loss/train":             train_loss,
                "loss/val":               val_loss,
                "loss/train_bc":          bc_loss,
                "loss/train_aux":         aux_loss,
                "loss/val_bc":            val_bc,
                "loss/val_aux":           val_aux,
                # Gradient health
                "train/grad_norm":        grad_norm,
                "train/lr_main":          lr_main,
                "train/lr_visual":        lr_visual,
                # RTG weight stats (sanity-check reward weighting)
                "train/rtg_mean":         np.mean(rtg_means),
                "train/rtg_min":          np.mean(rtg_mins),
                "train/rtg_max":          np.mean(rtg_maxs),
            }
            log_dict.update(avg_enc_norms)   # grad/enc_agent, grad/lstm, ...
            log_dict.update(avg_diag)        # diagnostics/* and metrics/*
            # Use global step for everything so W&B X-axis aligns correctly
            epoch_step = epoch * len(train_loader)
            wandb.log(log_dict, step=epoch_step)

        # ── Checkpointing ─────────────────────────────────────────── #
        ckpt = {
            "epoch":      epoch,
            "state_dict": policy.state_dict(),
            "val_loss":   float(val_loss),
            "train_loss": float(train_loss),
            "args":       vars(args),
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
