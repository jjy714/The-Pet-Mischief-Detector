from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights

# Output dimensions — referenced by model.py and roi_pool.py
FEAT_CHANNELS   = 256   # ResNet18 layer3 output channel count
GLOBAL_CHANNELS = 512   # ResNet18 avgpool output size
FEAT_STRIDE     = 16    # spatial downsampling from input to layer3 feature map


class ResNetBackbone(nn.Module):
    """
    Frozen ResNet18 feature extractor split into two stages:

      stem + layer1-3 → spatial feature map  (B, 256, H/16, W/16)
      layer4 + avgpool → global scene vector  (B, 512)

    All parameters are frozen; no gradients flow through this module.
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in resnet.parameters():
            p.requires_grad = False

        # Stage 1 — produces spatial feature map for RoI pooling
        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,   # (B, 256, H/16, W/16)
        )
        # Stage 2 — continues to a compact global descriptor
        self.stage2 = nn.Sequential(
            resnet.layer4,   # (B, 512, H/32, W/32)
            resnet.avgpool,  # (B, 512, 1, 1)
        )

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised tensor.

        Returns:
            feat_map:    (B, 256, H/16, W/16) — for RoI pooling per node.
            global_feat: (B, 512)             — for scene-level feature.
        """
        feat_map    = self.stage1(x)
        global_feat = self.stage2(feat_map).flatten(1)   # (B, 512)
        return feat_map, global_feat
