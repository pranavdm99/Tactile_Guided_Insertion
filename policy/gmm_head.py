"""
Gaussian Mixture Model action head for BC-RNN.

Action layout (7-D):
    dims 0:6  — [Δx, Δy, Δz, Δroll, Δpitch, Δyaw]   → K-component GMM
    dim   6   — gripper ∈ {-1.0 open, +1.0 closed}   → binary BCE

During training, `nll_loss` returns the reward-weighted negative log-likelihood.
During inference,  `act`    returns the mean of the highest-weight GMM component
                             plus the thresholded gripper action.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GMMHead(nn.Module):
    """
    K-mode Gaussian Mixture Model over the pose action space,
    with a separate binary head for the gripper.

    Args:
        input_dim:  Dimension of the LSTM hidden state fed in.
        action_dim: Pose action dimensionality (default 6, i.e. dx/dy/dz/dR/dP/dY).
        num_modes:  Number of GMM components K (default 3).
        min_std:    Lower clamp on predicted standard deviations for stability.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int = 6,
        num_modes:  int = 3,
        min_std:   float = 0.01,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_modes  = num_modes
        self.min_std    = min_std

        # Shared trunk projects LSTM features into a compact space
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),       nn.ReLU(inplace=True),
        )

        # Per-component Gaussian parameters
        self.mean_head       = nn.Linear(256, num_modes * action_dim)
        self.log_std_head    = nn.Linear(256, num_modes * action_dim)
        # Un-normalised log-mixture weights (softmax applied during forward)
        self.log_weight_head = nn.Linear(256, num_modes)

        # Binary gripper head (logit → sign at inference, BCE at training)
        self.gripper_head = nn.Linear(256, 1)

    # ─────────────────────────────────────────────────── #
    #  Internal forward: returns raw GMM parameters       #
    # ─────────────────────────────────────────────────── #

    def _forward(self, h: torch.Tensor):
        """
        Args:
            h: (*, input_dim)

        Returns:
            means:         (*, K, action_dim)
            stds:          (*, K, action_dim)  — positive, clamped
            weights:       (*, K)              — normalised mixture weights
            gripper_logit: (*, 1)
        """
        shape = h.shape[:-1]                                       # (*,)
        feat  = self.trunk(h)                                      # (*, 256)

        means   = self.mean_head(feat).reshape(*shape, self.num_modes, self.action_dim)
        log_std = self.log_std_head(feat).reshape(*shape, self.num_modes, self.action_dim)
        stds    = log_std.exp().clamp(min=self.min_std)

        weights       = F.softmax(self.log_weight_head(feat), dim=-1)  # (*, K)
        gripper_logit = self.gripper_head(feat)                        # (*, 1)

        return means, stds, weights, gripper_logit

    # ─────────────────────────────────────────────────── #
    #  Training: reward-weighted NLL                      #
    # ─────────────────────────────────────────────────── #

    def nll_loss(
        self,
        h:           torch.Tensor,
        actions:     torch.Tensor,
        rtg_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute the reward-weighted negative log-likelihood of the demonstrated
        actions under the predicted GMM + binary gripper distribution.

        Args:
            h:           (B, T, input_dim) — LSTM output sequence.
            actions:     (B, T, 7)         — ground-truth actions.
            rtg_weights: (B, T)            — Return-to-Go weights in [0.1, 1.0].
                                             If None, uniform weighting is used.

        Returns:
            Scalar loss.
        """
        pose_gt    = actions[..., :self.action_dim]   # (B, T, 6)
        gripper_gt = actions[..., self.action_dim:]   # (B, T, 1)  values ∈ {-1, +1}

        means, stds, weights, gripper_logit = self._forward(h)

        # ── Pose: GMM log-probability ──────────────────────────────────── #
        # Normal log-prob over each component: (B, T, K, action_dim)
        log_p = torch.distributions.Normal(means, stds).log_prob(
            pose_gt.unsqueeze(-2)
        )
        log_p = log_p.sum(-1)                          # sum over action dims: (B, T, K)
        # Weight by mixture coefficients, then marginalise via logsumexp:
        log_p = log_p + torch.log(weights.clamp(min=1e-8))
        log_prob_gmm = torch.logsumexp(log_p, dim=-1)  # (B, T)

        # ── Gripper: binary cross-entropy ──────────────────────────────── #
        # Map {-1, +1} → {0, 1}
        gripper_target = (gripper_gt + 1.0) / 2.0
        gripper_bce = F.binary_cross_entropy_with_logits(
            gripper_logit, gripper_target, reduction="none"
        ).squeeze(-1)                                  # (B, T)

        per_step_loss = -log_prob_gmm + gripper_bce   # (B, T)

        # ── Apply Return-to-Go weights ─────────────────────────────────── #
        if rtg_weights is not None:
            per_step_loss = per_step_loss * rtg_weights

        return per_step_loss.mean()

    # ─────────────────────────────────────────────────── #
    #  Inference: deterministic action                    #
    # ─────────────────────────────────────────────────── #

    @torch.no_grad()
    def act(self, h: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action: mean of the highest-weight GMM component,
        concatenated with the thresholded gripper action.

        Args:
            h: (B, input_dim) — typically B=1 during rollout.

        Returns:
            (B, 7) actions — pose (6) + gripper ∈ {-1.0, +1.0}.
        """
        means, _, weights, gripper_logit = self._forward(h)

        # Index of the highest-weight component: (B,)
        best = weights.argmax(dim=-1)

        # Gather the corresponding mean: (B, action_dim)
        idx  = best.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.action_dim)
        pose = means.gather(1, idx).squeeze(1)        # (B, action_dim)

        # Gripper: threshold logit to {-1, +1}
        gripper = torch.where(gripper_logit >= 0.0,
                              torch.ones_like(gripper_logit),
                              -torch.ones_like(gripper_logit))  # (B, 1)

        return torch.cat([pose, gripper], dim=-1)     # (B, 7)

    # ─────────────────────────────────────────────────── #
    #  Diagnostics (validation / W&B monitoring)          #
    # ─────────────────────────────────────────────────── #

    @torch.no_grad()
    def diagnostics(self, h: torch.Tensor, actions: torch.Tensor) -> dict:
        """
        Compute interpretable diagnostic metrics over a batch for W&B logging.
        Should be called during the validation pass (no gradients needed).

        Args:
            h:       (B, T, input_dim) LSTM output.
            actions: (B, T, 7)         ground-truth actions.

        Returns:
            dict with scalar diagnostic values.
        """
        means, stds, weights, gripper_logit = self._forward(h)

        pose_gt    = actions[..., :self.action_dim]  # (B, T, 6)
        gripper_gt = actions[..., self.action_dim:]  # (B, T, 1)

        # ── GMM std diagnostics ───────────────────────────────────────── #
        # Mean std across all modes and action dims — tracks confidence
        mean_std = stds.mean().item()

        # Per-mode mean std (K scalars) — reveals if one mode collapsed
        per_mode_std = stds.mean(dim=(0, 1, -1))  # (K,)

        # ── Mixture weight diagnostics ────────────────────────────────── #
        # Weight entropy: high = diverse, low = mode collapse
        eps = 1e-8
        weight_entropy = -(weights * (weights + eps).log()).sum(-1).mean().item()

        # Dominant mode fraction: how often is one mode overwhelming?
        dominant_mode  = weights.argmax(dim=-1)                  # (B, T)
        dominant_frac  = [(dominant_mode == k).float().mean().item()
                          for k in range(self.num_modes)]

        # ── Pose: best-mode MAE ───────────────────────────────────────── #
        # Use highest-weight mode mean as predicted action, compute MAE
        best_idx  = weights.argmax(dim=-1, keepdim=True)         # (B, T, 1)
        best_idx  = best_idx.unsqueeze(-1).expand(*best_idx.shape, self.action_dim)
        pred_pose = means.gather(-2, best_idx).squeeze(-2)       # (B, T, 6)
        pose_mae  = (pred_pose - pose_gt).abs().mean().item()

        # Per-DOF MAE: [dx, dy, dz, droll, dpitch, dyaw]
        dof_names = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]
        per_dof_mae = {
            f"metrics/mae_{name}": (pred_pose - pose_gt).abs().mean(dim=(0, 1))[i].item()
            for i, name in enumerate(dof_names)
        }

        # ── Gripper accuracy ──────────────────────────────────────────── #
        pred_gripper = (gripper_logit >= 0.0).float() * 2.0 - 1.0  # {-1, +1}
        gripper_acc  = (pred_gripper == gripper_gt).float().mean().item()

        # Fraction of time gripper is predicted closed (sanity check)
        gripper_closed_pred = (gripper_logit >= 0.0).float().mean().item()
        gripper_closed_gt   = (gripper_gt > 0).float().mean().item()

        result = {
            "diagnostics/gmm_mean_std":       mean_std,
            "diagnostics/gmm_weight_entropy": weight_entropy,
            "diagnostics/pose_mae":           pose_mae,
            "metrics/gripper_accuracy":       gripper_acc,
            "diagnostics/gripper_closed_pred_frac": gripper_closed_pred,
            "diagnostics/gripper_closed_gt_frac":   gripper_closed_gt,
        }
        result.update(per_dof_mae)
        for k, frac in enumerate(dominant_frac):
            result[f"diagnostics/gmm_mode{k}_dominant_frac"] = frac
        for k, s in enumerate(per_mode_std.tolist()):
            result[f"diagnostics/gmm_mode{k}_mean_std"] = s

        return result

