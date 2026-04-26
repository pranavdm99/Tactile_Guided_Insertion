"""
Encoders for the BC-RNN multimodal policy.

  VisualEncoder  — ResNet-18 (ImageNet pre-trained) + linear projection
  TactileEncoder — Shared shallow CNN + Spatial Softmax
  ProprioEncoder — 2-layer MLP over task-space state

All encoders operate on a flat (B, ...) batch; the BC-RNN policy is
responsible for reshaping (B, T, ...) inputs before calling these.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────── #
#  Utility: Quaternion (x, y, z, w) → 6-D Rotation (Zhou et al. 2019)        #
# ─────────────────────────────────────────────────────────────────────────── #

def quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """
    Maps unit quaternions in (x, y, z, w) convention to the 6-D rotation
    representation (first two *columns* of the rotation matrix, stacked flat).

    Args:
        q: (*, 4) float tensor.

    Returns:
        (*, 6) float tensor.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # First column of R
    col0 = torch.stack([
        1.0 - 2.0 * (y**2 + z**2),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    ], dim=-1)   # (*, 3)

    # Second column of R
    col1 = torch.stack([
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x**2 + z**2),
        2.0 * (y * z + x * w),
    ], dim=-1)   # (*, 3)

    return torch.cat([col0, col1], dim=-1)   # (*, 6)


# ─────────────────────────────────────────────────────────────────────────── #
#  Spatial Softmax                                                             #
# ─────────────────────────────────────────────────────────────────────────── #

class SpatialSoftmax(nn.Module):
    """
    Converts a (B, C, H, W) feature map to (B, C*2) expected (x, y) keypoint
    coordinates in the range [-1, 1].

    This preserves spatial structure while compressing the feature map to a
    compact, interpretable keypoint representation — effectively telling the
    policy *where* each learned feature is located on the tactile pad.

    Args:
        num_channels: Number of feature channels C (= number of keypoints).
        height:       Spatial height H of the feature map.
        width:        Spatial width  W of the feature map.
        temperature:  Softmax temperature (lower = sharper focus).
    """

    def __init__(
        self,
        num_channels: int,
        height: int,
        width: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature

        # Pre-compute position grids — shape (1, 1, H*W)
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C*2) keypoint coordinates.
        """
        B, C, H, W = x.shape
        flat = x.reshape(B, C, -1)                                # (B, C, H*W)
        attn = F.softmax(flat / self.temperature, dim=-1)

        expected_x = (attn * self.pos_x).sum(-1)                  # (B, C)
        expected_y = (attn * self.pos_y).sum(-1)                  # (B, C)

        return torch.cat([expected_x, expected_y], dim=-1)        # (B, C*2)


# ─────────────────────────────────────────────────────────────────────────── #
#  Visual Encoder — one per camera (agentview and wrist)                      #
# ─────────────────────────────────────────────────────────────────────────── #

class VisualEncoder(nn.Module):
    """
    ResNet-18 backbone (ImageNet pre-trained) with a linear projection head.

    The backbone is frozen by default for the first few training epochs
    (warm-up), then fine-tuned at a low LR via `.unfreeze()`.

    Args:
        out_dim:          Output feature dimension.
        freeze_backbone:  If True, backbone weights are non-trainable initially.
    """

    def __init__(self, out_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()

        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip classification head; backbone ends with AdaptiveAvgPool → (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.project = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

        # Register ImageNet stats for on-the-fly normalisation
        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).reshape(1, 3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32).reshape(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std",  std)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze(self):
        """Unfreeze backbone for fine-tuning (call after warm-up epochs)."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def _normalise(self, img: torch.Tensor) -> torch.Tensor:
        """Expects (B, 3, H, W) float32 in [0, 1]."""
        return (img - self.img_mean) / self.img_std

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) float32 in [0, 1].

        Returns:
            (B, out_dim) float32.
        """
        x = self._normalise(img)
        feat = self.backbone(x).flatten(1)    # (B, 512)
        return self.project(feat)             # (B, out_dim)


# ─────────────────────────────────────────────────────────────────────────── #
#  Tactile Encoder — shared between left and right fingers                    #
# ─────────────────────────────────────────────────────────────────────────── #

class TactileEncoder(nn.Module):
    """
    Shallow CNN + Spatial Softmax for FOTS-synthesised tactile RGB images.

    Weights are *shared* between the left and right fingers: the same sensor
    geometry means the same features are meaningful on both sides, and sharing
    halves the parameter count while doubling the effective tactile training
    signal.

    Args:
        in_h:           Image height  (96 for our 96×128 tactile renders).
        in_w:           Image width   (128).
        num_keypoints:  Number of Spatial-Softmax keypoints (output = 2×this).
    """

    def __init__(self, in_h: int = 96, in_w: int = 128, num_keypoints: int = 32):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.out_dim = num_keypoints * 2          # x + y per keypoint

        self.cnn = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # H/2, W/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, kernel_size=3, padding=1),
        )

        # After MaxPool2d(2): spatial size = (in_h // 2, in_w // 2) = (48, 64)
        self.spatial_softmax = SpatialSoftmax(num_keypoints, in_h // 2, in_w // 2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: (B, 3, H, W) float32 in [0, 1].

        Returns:
            (B, num_keypoints * 2) float32 keypoint coordinates.
        """
        feat = self.cnn(img)                     # (B, num_keypoints, H/2, W/2)
        return self.spatial_softmax(feat)        # (B, num_keypoints * 2)


# ─────────────────────────────────────────────────────────────────────────── #
#  Proprioception Encoder                                                      #
# ─────────────────────────────────────────────────────────────────────────── #

class ProprioEncoder(nn.Module):
    """
    2-layer MLP over the task-space proprioceptive state vector.

    Input layout (11-D):
        robot0_eef_pos      (3)  XYZ in robot base frame
        robot0_eef_quat     (4) → 6-D rotation (converted externally)
        robot0_gripper_qpos (2)  finger widths

    The conversion quat → 6D is done by the BCRNNPolicy before this call,
    so in_dim = 3 + 6 + 2 = 11.

    Args:
        in_dim:  Input dimensionality (default 11).
        out_dim: Output feature dimensionality (default 64).
    """

    def __init__(self, in_dim: int = 11, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim).

        Returns:
            (B, out_dim).
        """
        return self.net(x)
