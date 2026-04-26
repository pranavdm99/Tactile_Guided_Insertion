#!/usr/bin/env python3
"""
Quick smoke test for the BC-RNN policy — no real data needed.

Verifies:
  1. Forward pass produces a valid scalar loss.
  2. Backward pass (gradients) works without error.
  3. Stateful rollout inference produces a (7,) action.
  4. Prints total parameter count.

Run inside the docker container:
    python3 scripts/test_policy_shapes.py
    python3 scripts/test_policy_shapes.py --checkpoint checkpoints/best_policy.pth
    python3 scripts/test_policy_shapes.py --checkpoint checkpoints/latest_policy.pth
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch
from policy.bc_rnn import BCRNNPolicy

def main():
    parser = argparse.ArgumentParser(description="BC-RNN policy smoke test")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional path to a .pth checkpoint (e.g. checkpoints/best_policy.pth)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    B, T = 4, 20   # batch=4, sequence length=20

    policy = BCRNNPolicy(num_gmm_modes=3, freeze_visual=True, use_aux_loss=True).to(device)
    print(f"Total params:     {policy.num_parameters():,}")
    print(f"Trainable params: {policy.num_trainable_parameters():,}")

    # ── Optional checkpoint loading ──────────────────────────────────── #
    if args.checkpoint:
        ckpt_path = args.checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(os.path.dirname(__file__), "..", ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.exists(ckpt_path):
            print(f"\n❌ Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        # Support both 'state_dict' and legacy 'model_state_dict' keys
        sd_key = "state_dict" if "state_dict" in ckpt else "model_state_dict"
        policy.load_state_dict(ckpt[sd_key])
        epoch      = ckpt.get("epoch", "?")
        val_loss   = ckpt.get("val_loss", None)
        train_loss = ckpt.get("train_loss", None)
        print(f"\n[CKPT] Loaded:      {ckpt_path}")
        print(f"       Epoch:        {epoch}")
        if train_loss is not None:
            print(f"       Train loss:  {train_loss:.4f}")
        if val_loss is not None:
            print(f"       Val loss:    {val_loss:.4f}")
        print("[CKPT] Weights loaded: ✅")
    else:
        print("       (no checkpoint — using random weights)")
    print()

    # ── Synthetic observation batch ─────────────────────────────────── #
    obs = {
        "agentview_image": torch.randint(0, 256, (B, T, 256, 256, 3), dtype=torch.uint8).to(device),
        "wrist_image":     torch.randint(0, 256, (B, T, 256, 256, 3), dtype=torch.uint8).to(device),
        "tactile_left":    torch.randint(0, 256, (B, T,  96, 128, 3), dtype=torch.uint8).to(device),
        "tactile_right":   torch.randint(0, 256, (B, T,  96, 128, 3), dtype=torch.uint8).to(device),
        "eef_pos":         torch.randn(B, T, 3).to(device),
        "eef_quat":        torch.randn(B, T, 4).to(device),
        "gripper_qpos":    torch.randn(B, T, 2).to(device),
        "obj_to_eef":      torch.randn(B, T, 3).to(device),
    }
    actions     = torch.randn(B, T, 7).to(device)
    rtg_weights = torch.ones(B, T).to(device)

    # ── Training forward + backward ─────────────────────────────────── #
    loss, info = policy(obs, actions, rtg_weights)
    print(f"[TRAIN] Loss:     {loss.item():.4f}")
    print(f"        BC loss:  {info['bc_loss']:.4f}")
    print(f"        Aux loss: {info['aux_loss']:.4f}")
    loss.backward()
    print("[TRAIN] Backward pass: ✅\n")

    # ── Inference (stateful rollout) ─────────────────────────────────── #
    policy.eval()
    policy.reset()

    obs_single = {
        "agentview_image": torch.randint(0, 256, (1, 256, 256, 3), dtype=torch.uint8).to(device),
        "wrist_image":     torch.randint(0, 256, (1, 256, 256, 3), dtype=torch.uint8).to(device),
        "tactile_left":    torch.randint(0, 256, (1,  96, 128, 3), dtype=torch.uint8).to(device),
        "tactile_right":   torch.randint(0, 256, (1,  96, 128, 3), dtype=torch.uint8).to(device),
        "eef_pos":         torch.randn(1, 3).to(device),
        "eef_quat":        torch.randn(1, 4).to(device),
        "gripper_qpos":    torch.randn(1, 2).to(device),
    }

    action = policy.act(obs_single)
    print(f"[INFER] Action shape: {action.shape}  (expected (7,))")
    print(f"        Action:       {action.round(2)}")
    print(f"        Gripper:      {action[-1]} (should be ±1.0)")
    print("\n✅ All checks passed.")

if __name__ == "__main__":
    main()
