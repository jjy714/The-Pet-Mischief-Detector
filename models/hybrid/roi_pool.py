from __future__ import annotations

import torch
import torchvision.ops as tv_ops

from models.hybrid.backbone import FEAT_CHANNELS, FEAT_STRIDE
from schema.Data import Detection

# RoI align outputs a 7×7 spatial patch; global avg-pool collapses it to (C,)
_ROI_SPATIAL = 7
ROI_DIM = FEAT_CHANNELS   # 256 — raw visual feature dim before roi_proj in the model


def extract_roi_features(
    feat_map: torch.Tensor,
    detections: list[Detection],
    img_size: int = 640,
) -> torch.Tensor:
    """
    Extract a fixed-size visual feature for each detection via RoI Align.

    The feature map (from ResNet18 layer3) has spatial stride FEAT_STRIDE=16,
    so pixel-space bounding boxes are divided by that stride implicitly via
    the spatial_scale parameter.

    Args:
        feat_map:   (1, FEAT_CHANNELS, H', W') — layer3 output for a single image.
        detections: YOLO detections with normalised bbox coords in [0, 1].
        img_size:   Pixel size the image was resized to before the backbone (640).

    Returns:
        (N, ROI_DIM) tensor — one avg-pooled visual vector per detection.
        Returns a zero tensor of shape (0, ROI_DIM) when detections is empty.
    """
    if not detections:
        return torch.zeros((0, ROI_DIM), dtype=torch.float, device=feat_map.device)

    # Convert normalised bbox coords → pixel coords in the resized image
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
    )  # (N, 4)

    # roi_align: expects list of (K_i, 4) boxes, one per image in the batch.
    # spatial_scale maps pixel coords to feature-map coords (1 / stride).
    roi_out = tv_ops.roi_align(
        feat_map,                        # (1, C, H', W')
        [boxes],                         # single image → list of length 1
        output_size=_ROI_SPATIAL,
        spatial_scale=1.0 / FEAT_STRIDE,
        aligned=True,
    )  # (N, C, 7, 7)

    return roi_out.mean(dim=[2, 3])      # (N, C) — global average pool over 7×7
