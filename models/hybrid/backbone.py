from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights

# output dimensions referenced by model.py and roi_pool.py
FEAT_CHANNELS   = 256
GLOBAL_CHANNELS = 512
FEAT_STRIDE     = 16


# frozen ResNet18 split into a spatial feature map stage and a global descriptor stage
class ResNetBackbone(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        resnet = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in resnet.parameters():
            p.requires_grad = False

        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.stage2 = nn.Sequential(
            resnet.layer4,
            resnet.avgpool,
        )

    ## extract spatial feature map and global scene vector from an ImageNet-normalised tensor
    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map    = self.stage1(x)
        global_feat = self.stage2(feat_map).flatten(1)
        return feat_map, global_feat
