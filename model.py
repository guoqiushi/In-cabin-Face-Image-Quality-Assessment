# model.py
# Multi-head Face Quality Assessment model with MobileNetV3 backbone
# Supports auxiliary heads for:
#   - face recognition (embedding)
#   - gaze estimation (yaw/pitch or 3D vector)
#   - keypoint detection (2D landmark coordinates)
#
# Output:
#   dict with keys: "quality", "fr_emb", "gaze", "kpts"
#
# Requirements:
#   pip install torch torchvision

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
except Exception as e:
    raise ImportError("torchvision is required for MobileNetV3 backbone. Install torchvision.") from e


@dataclass
class HeadConfig:
    # Face recognition
    fr_embedding_dim: int = 512  # output embedding dimension

    # Gaze estimation
    gaze_out_dim: int = 2  # (yaw, pitch) by default. set 3 for 3D gaze vector.

    # Keypoints
    num_keypoints: int = 68  # number of landmarks
    kpt_out_type: str = "coord"  # "coord" only (N, K, 2). Heatmap head not included for simplicity.

    # Quality
    quality_act: str = "sigmoid"  # "sigmoid" or "none"


class MLPHead(nn.Module):
    """
    Generic MLP head for global features.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        act: str = "relu",
        out_act: str = "none",
    ):
        super().__init__()
        act_layer = nn.ReLU(inplace=True) if act == "relu" else nn.SiLU(inplace=True)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_layer,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_act = out_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.out_act == "sigmoid":
            y = torch.sigmoid(y)
        elif self.out_act == "tanh":
            y = torch.tanh(y)
        return y


class CoordKeypointHead(nn.Module):
    """
    Predict 2D keypoint coordinates from global features.
    Output shape: (N, K, 2), in normalized coordinate space by default.
    You can map to pixel coords outside using image size.
    """
    def __init__(
        self,
        in_dim: int,
        num_keypoints: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        coord_act: str = "tanh",
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_keypoints * 2),
        )
        self.coord_act = coord_act  # "tanh" -> [-1,1], "sigmoid" -> [0,1], "none" -> raw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x).view(-1, self.num_keypoints, 2)
        if self.coord_act == "tanh":
            y = torch.tanh(y)
        elif self.coord_act == "sigmoid":
            y = torch.sigmoid(y)
        return y


class MultiHeadFIQA(nn.Module):
    """
    MobileNetV3-based multi-head model:
      - quality: scalar in [0, 1] (default sigmoid)
      - face recognition: L2-normalized embedding
      - gaze: yaw/pitch (or gaze vector)
      - keypoints: normalized 2D coords
    """
    def __init__(
        self,
        backbone: str = "small",  # "small" or "large"
        pretrained: bool = False,
        in_channels: int = 3,
        head_cfg: HeadConfig = HeadConfig(),
        feat_dim: int = 256,  # shared feature dim after projection
        dropout: float = 0.1,
    ):
        super().__init__()
        assert backbone in ("small", "large")

        # Backbone
        if backbone == "small":
            base = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        else:
            base = mobilenet_v3_large(weights="DEFAULT" if pretrained else None)

        # Replace first conv if input channels != 3
        if in_channels != 3:
            self._replace_first_conv(base, in_channels)

        # Use MobileNetV3 features (conv stages)
        self.backbone = base.features

        # Infer backbone output channels by forwarding a dummy tensor (safe for scripting)
        out_ch = self._infer_backbone_out_channels(in_channels=in_channels)

        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.SiLU(inplace=True),
        )

        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=dropout)

        # Heads
        self.head_cfg = head_cfg

        # Quality head: scalar
        self.quality_head = MLPHead(
            in_dim=feat_dim,
            out_dim=1,
            hidden_dim=max(128, feat_dim // 2),
            dropout=dropout,
            act="silu",
            out_act=head_cfg.quality_act,
        )

        # Face recognition embedding head
        self.fr_head = MLPHead(
            in_dim=feat_dim,
            out_dim=head_cfg.fr_embedding_dim,
            hidden_dim=max(256, feat_dim),
            dropout=dropout,
            act="silu",
            out_act="none",
        )

        # Gaze head
        self.gaze_head = MLPHead(
            in_dim=feat_dim,
            out_dim=head_cfg.gaze_out_dim,
            hidden_dim=max(128, feat_dim // 2),
            dropout=dropout,
            act="silu",
            out_act="none",
        )

        # Keypoint head (coords)
        if head_cfg.kpt_out_type != "coord":
            raise ValueError("Only kpt_out_type='coord' is supported in this file.")
        self.kpt_head = CoordKeypointHead(
            in_dim=feat_dim,
            num_keypoints=head_cfg.num_keypoints,
            hidden_dim=max(256, feat_dim),
            dropout=dropout,
            coord_act="tanh",  # [-1,1] normalized; change to sigmoid for [0,1]
        )

    @torch.no_grad()
    def _infer_backbone_out_channels(self, in_channels: int) -> int:
        # Try on CPU to avoid GPU surprises
        self.backbone.eval()
        x = torch.zeros(1, in_channels, 224, 224)
        y = self.backbone(x)
        return int(y.shape[1])

    def _replace_first_conv(self, model, in_channels: int) -> None:
        # MobileNetV3 first block is a Conv2dNormActivation
        # We replace its Conv2d to accept in_channels.
        first = model.features[0]
        conv = first[0]  # Conv2d
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False,
            padding_mode=conv.padding_mode,
        )

        # If original had 3 channels, roughly initialize by channel-mean
        with torch.no_grad():
            if conv.weight.shape[1] == 3:
                w = conv.weight
                if in_channels == 1:
                    new_conv.weight.copy_(w.mean(dim=1, keepdim=True))
                else:
                    # Repeat / trim with mean fallback
                    mean_w = w.mean(dim=1, keepdim=True)
                    rep = mean_w.repeat(1, in_channels, 1, 1)
                    # If we can, also copy the first 3 channels for stability
                    rep[:, :3].copy_(w)
                    new_conv.weight.copy_(rep / rep.shape[1] * 3.0)
            else:
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        first[0] = new_conv

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns shared global feature vector of shape (N, feat_dim).
        """
        f = self.backbone(x)          # (N, C, H, W)
        f = self.proj(f)              # (N, feat_dim, H, W)
        f = self.pool(f)              # (N, feat_dim, 1, 1)
        f = self.flatten(f)           # (N, feat_dim)
        f = self.dropout(f)
        return f

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
        normalize_fr: bool = True,
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: input image tensor (N, C, H, W), recommended normalized to ImageNet stats if using pretrained.
            return_dict: if True, returns a dict; else returns a tuple.
            normalize_fr: L2-normalize face recognition embedding.

        Returns:
            dict or tuple:
              quality: (N, 1) in [0,1] if sigmoid, else raw
              fr_emb: (N, D) L2-normalized if normalize_fr
              gaze: (N, G)
              kpts: (N, K, 2) normalized coords (default [-1,1])
        """
        feat = self.extract_feat(x)

        quality = self.quality_head(feat)  # (N, 1)
        fr_emb = self.fr_head(feat)        # (N, D)
        if normalize_fr:
            fr_emb = F.normalize(fr_emb, p=2, dim=1, eps=1e-12)

        gaze = self.gaze_head(feat)        # (N, G)
        kpts = self.kpt_head(feat)         # (N, K, 2)

        if return_dict:
            return {
                "quality": quality,
                "fr_emb": fr_emb,
                "gaze": gaze,
                "kpts": kpts,
            }
        return quality, fr_emb, gaze, kpts


if __name__ == "__main__":
    # Quick sanity check
    model = MultiHeadFIQA(backbone="small", pretrained=False, in_channels=3)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    for k, v in y.items():
        print(k, tuple(v.shape))

