"""
BC-RNN Policy — Multimodal Behavior Cloning with LSTM temporal backbone.

Architecture:
    agentview_image  → VisualEncoder  → (256,)  ┐
    wrist_image      → VisualEncoder  → (256,)  │
    tactile_left     → TactileEncoder → (64,)   ├─ cat → (704,) → LSTM → GMMHead
    tactile_right    → TactileEncoder → (64,)   │
    eef + rot + grip → ProprioEncoder → (64,)   ┘

Training:   policy.forward(obs, actions, rtg_weights) → loss
Inference:  policy.reset(); action = policy.act(obs_single)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VisualEncoder, TactileEncoder, ProprioEncoder, quat_to_6d
from .gmm_head import GMMHead


class BCRNNPolicy(nn.Module):
    """
    Full BC-RNN policy (encoders + LSTM + GMM head + optional aux observer).

    Dimension constants (edit here to change the whole network):
        VISUAL_DIM  = 256   per camera
        TACTILE_DIM = 64    per finger (32 keypoints × 2 coords)
        PROPRIO_DIM = 64
        FUSION_DIM  = 256 + 256 + 64 + 64 + 64 = 704
        LSTM_HIDDEN = 1024
        LSTM_LAYERS = 2

    Args:
        num_gmm_modes:  K for the GMM action head (default 3).
        freeze_visual:  Freeze ResNet backbones for the first few epochs.
        use_aux_loss:   Attach an auxiliary head that predicts object_to_eef_pos
                        (uses ground-truth from HDF5 only during *training*).
    """

    # ── Architecture constants ──────────────────────────────────────── #
    VISUAL_DIM  = 256
    TACTILE_DIM = 64   # 32 keypoints × 2
    PROPRIO_DIM = 64
    FUSION_DIM  = VISUAL_DIM + VISUAL_DIM + TACTILE_DIM + TACTILE_DIM + PROPRIO_DIM  # 704
    LSTM_HIDDEN = 1024
    LSTM_LAYERS = 2

    def __init__(
        self,
        num_gmm_modes: int = 3,
        freeze_visual: bool = True,
        use_aux_loss:  bool = True,
    ):
        super().__init__()
        self.use_aux_loss = use_aux_loss

        # ── Encoders ─────────────────────────────────────────────────── #
        self.enc_agent  = VisualEncoder(out_dim=self.VISUAL_DIM,  freeze_backbone=freeze_visual)
        self.enc_wrist  = VisualEncoder(out_dim=self.VISUAL_DIM,  freeze_backbone=freeze_visual)

        # Tactile: left and right share the same CNN + Spatial Softmax weights
        self.enc_tac    = TactileEncoder(in_h=96, in_w=128, num_keypoints=32)

        self.enc_proprio = ProprioEncoder(in_dim=11, out_dim=self.PROPRIO_DIM)

        # ── Temporal backbone ─────────────────────────────────────────── #
        self.lstm = nn.LSTM(
            input_size=self.FUSION_DIM,
            hidden_size=self.LSTM_HIDDEN,
            num_layers=self.LSTM_LAYERS,
            batch_first=True,
            dropout=0.1,
        )

        # ── Action head ───────────────────────────────────────────────── #
        self.action_head = GMMHead(
            input_dim=self.LSTM_HIDDEN,
            action_dim=6,
            num_modes=num_gmm_modes,
        )

        # ── Auxiliary observer head (training only) ───────────────────── #
        # Predicts object_to_eef_pos from the fused features, forcing the
        # visual+tactile encoders to be spatially aware of the peg location
        # without leaking object_pos into the policy at test time.
        if use_aux_loss:
            self.observer_head = nn.Sequential(
                nn.Linear(self.FUSION_DIM, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 3),
            )

        # ── Inference state (stateful LSTM hidden) ─────────────────────── #
        self._hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    # ──────────────────────────────────────────────────────────────────── #
    #  Image preprocessing                                                 #
    # ──────────────────────────────────────────────────────────────────── #

    @staticmethod
    def _prep_img(img: torch.Tensor) -> torch.Tensor:
        """
        Converts an image batch with a time dimension from HDF5 format to the
        channels-first float format expected by the encoders.

        Args:
            img: (B, T, H, W, 3) uint8 (or float32 [0,1]).

        Returns:
            (B*T, 3, H, W) float32 in [0, 1].
        """
        B, T = img.shape[:2]
        img  = img.reshape(B * T, *img.shape[2:])           # (B*T, H, W, 3)
        img  = img.permute(0, 3, 1, 2).contiguous()         # (B*T, 3, H, W)
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    # ──────────────────────────────────────────────────────────────────── #
    #  Encoder pass                                                        #
    # ──────────────────────────────────────────────────────────────────── #

    def _encode(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode a full (B, T, ...) observation dict into a fused feature tensor.

        Args:
            obs: Dict with keys
                    agentview_image     (B, T, H, W, 3)
                    wrist_image         (B, T, H, W, 3)
                    tactile_left        (B, T, H, W, 3)
                    tactile_right       (B, T, H, W, 3)
                    eef_pos             (B, T, 3)
                    eef_quat            (B, T, 4)
                    gripper_qpos        (B, T, 2)
                    obj_to_eef  (opt)   (B, T, 3)  — only used for aux loss

        Returns:
            fused:    (B, T, FUSION_DIM)
            obs_pred: (B, T, 3) predicted object_to_eef_pos, or None
        """
        B, T = obs["agentview_image"].shape[:2]

        # ── Visual streams ───────────────────────────────────────────── #
        av = self._prep_img(obs["agentview_image"])           # (B*T, 3, 256, 256)
        wc = self._prep_img(obs["wrist_image"])               # (B*T, 3, 256, 256)
        f_av = self.enc_agent(av).reshape(B, T, -1)           # (B, T, 256)
        f_wc = self.enc_wrist(wc).reshape(B, T, -1)           # (B, T, 256)

        # ── Tactile streams (shared encoder) ─────────────────────────── #
        tl   = self._prep_img(obs["tactile_left"])            # (B*T, 3, 96, 128)
        tr   = self._prep_img(obs["tactile_right"])           # (B*T, 3, 96, 128)
        f_tl = self.enc_tac(tl).reshape(B, T, -1)            # (B, T, 64)
        f_tr = self.enc_tac(tr).reshape(B, T, -1)            # (B, T, 64)

        # ── Proprioception ───────────────────────────────────────────── #
        eef_pos  = obs["eef_pos"].reshape(B * T, 3)           # (B*T, 3)
        eef_quat = obs["eef_quat"].reshape(B * T, 4).float()
        rot6d    = quat_to_6d(eef_quat)                       # (B*T, 6)
        gripper  = obs["gripper_qpos"].reshape(B * T, 2)      # (B*T, 2)

        proprio  = torch.cat([eef_pos, rot6d, gripper], dim=-1)  # (B*T, 11)
        f_prop   = self.enc_proprio(proprio).reshape(B, T, -1)   # (B, T, 64)

        # ── Fusion ───────────────────────────────────────────────────── #
        fused = torch.cat([f_av, f_wc, f_tl, f_tr, f_prop], dim=-1)  # (B, T, 704)

        # ── Auxiliary: predict object_to_eef from fusion features ─────── #
        obs_pred = None
        if self.use_aux_loss:
            obs_pred = self.observer_head(fused)               # (B, T, 3)

        return fused, obs_pred

    # ──────────────────────────────────────────────────────────────────── #
    #  Training forward pass                                               #
    # ──────────────────────────────────────────────────────────────────── #

    def forward(
        self,
        obs:         dict[str, torch.Tensor],
        actions:     torch.Tensor,
        rtg_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Full supervised training pass over a (B, T) sequence.

        Args:
            obs:         Dict of (B, T, ...) tensors.
            actions:     (B, T, 7) ground-truth actions.
            rtg_weights: (B, T) Return-to-Go weights ∈ [0.1, 1.0].

        Returns:
            total_loss: Scalar.
            info:       Dict with 'bc_loss' and 'aux_loss' for logging.
        """
        fused, obs_pred = self._encode(obs)                 # (B, T, 704)

        # Run LSTM + loss in float32: 1/σ² gradients from the GMM head can be
        # large enough to overflow float16 in lstm_out's gradient buffer, which
        # propagates inf back through LSTM weights and all upstream encoders.
        with torch.autocast("cuda", enabled=False):
            lstm_out, _ = self.lstm(fused.float())          # (B, T, 1024) f32
            bc_loss = self.action_head.nll_loss(
                lstm_out, actions.float(), rtg_weights
            )

        # ── Auxiliary observer loss ───────────────────────────────────── #
        aux_loss = torch.tensor(0.0, device=bc_loss.device)
        if self.use_aux_loss and obs_pred is not None and "obj_to_eef" in obs:
            obj_gt   = obs["obj_to_eef"].float()             # (B, T, 3)
            aux_loss = F.mse_loss(obs_pred, obj_gt)

        total_loss = bc_loss + 0.1 * aux_loss

        return total_loss, {
            "bc_loss":  bc_loss.item(),
            "aux_loss": aux_loss.item(),
        }

    # ──────────────────────────────────────────────────────────────────── #
    #  Inference — stateful single-step rollout                            #
    # ──────────────────────────────────────────────────────────────────── #

    def reset(self):
        """Reset LSTM hidden state. Call at the start of each rollout episode."""
        self._hidden = None

    @torch.no_grad()
    def act(self, obs_single: dict[str, torch.Tensor]) -> "np.ndarray":
        """
        Single-step inference. Maintains LSTM hidden state across calls.

        Args:
            obs_single: Dict of (1, ...) tensors — *no* time dimension.
                        Keys: agentview_image, wrist_image, tactile_left,
                              tactile_right, eef_pos, eef_quat, gripper_qpos.

        Returns:
            (7,) numpy float32 action array.
        """
        # Add fake T=1 dimension to match encoder expectations
        obs_seq = {k: v.unsqueeze(1) for k, v in obs_single.items()}

        fused, _ = self._encode(obs_seq)                    # (1, 1, 704)
        lstm_out, self._hidden = self.lstm(fused, self._hidden)  # (1, 1, 1024)

        h_t    = lstm_out[:, 0, :]                          # (1, 1024)
        action = self.action_head.act(h_t)                  # (1, 7)

        return action.squeeze(0).cpu().numpy()

    # ──────────────────────────────────────────────────────────────────── #
    #  Helpers                                                             #
    # ──────────────────────────────────────────────────────────────────── #

    def unfreeze_visual(self):
        """Unfreeze ResNet backbones for fine-tuning (call after warm-up)."""
        self.enc_agent.unfreeze()
        self.enc_wrist.unfreeze()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
