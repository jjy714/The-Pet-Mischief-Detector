from __future__ import annotations

import torch
import torchvision.ops as tv_ops

from models.hybrid.backbone import FEAT_CHANNELS, FEAT_STRIDE
from schema.Data import Detection

_ROI_SPATIAL = 7
ROI_DIM = FEAT_CHANNELS


## extract a fixed-size visual feature for each detection via RoI Align on the ResNet18 layer3 feature map
def extract_roi_features(
    feat_map: torch.Tensor,
    detections: list[Detection],
    img_size: int = 640,
) -> torch.Tensor:
    if not detections:
        return torch.zeros((0, ROI_DIM), dtype=torch.float, device=feat_map.device)

    boxes = torch.tensor(
        [
            [
                det.bbox.x_min * img_size,
                det.bbox.y_min * img_size,
                det.bbox.x_max * img_size,
                det.bbox.y_max * img_size,
            ]
            for det in detections
        ],
        dtype=torch.float,
        device=feat_map.device,
    )

    ### spatial_scale = 1 / FEAT_STRIDE maps pixel-space bbox coords to feature-map coords
    roi_out = tv_ops.roi_align(
        feat_map,
        [boxes],
        output_size=_ROI_SPATIAL,
        spatial_scale=1.0 / FEAT_STRIDE,
        aligned=True,
    )

    return roi_out.mean(dim=[2, 3])
