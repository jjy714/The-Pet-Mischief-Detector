from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
import cv2

from torch_geometric.data import Batch

from models.gnn.dataset import LABEL_NAMES
from models.hybrid.backbone import ResNetBackbone
from models.hybrid.graph_builder import build_static_graph
from models.hybrid.model import HybridMischiefModel
from models.hybrid.roi_pool import extract_roi_features
from schema.Data import Detection

_IMG_SIZE      = 640
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess_frame(frame: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """
    Convert a BGR OpenCV frame to a (1, 3, 640, 640) ImageNet-normalised tensor.

    Used both during cache pre-computation (07_train_hybrid.py) and during
    live inference (predict_image).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t   = TF.resize(t, [_IMG_SIZE, _IMG_SIZE], antialias=True)
    t   = TF.normalize(t, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return t.unsqueeze(0).to(device)           # (1, 3, 640, 640)


def load_hybrid_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[ResNetBackbone, HybridMischiefModel]:
    """
    Load frozen backbone and trained HybridMischiefModel from checkpoint.

    Returns:
        (backbone, model) — both in eval mode on the requested device.
    """
    backbone = ResNetBackbone().to(device)
    backbone.eval()

    model = HybridMischiefModel().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return backbone, model


def predict_image(
    backbone: ResNetBackbone,
    model: HybridMischiefModel,
    frame: np.ndarray,
    detections: list[Detection],
    device: str = "cpu",
) -> str:
    """
    Predict mischief risk for a single static frame.

    Args:
        backbone:   Frozen ResNetBackbone (eval mode).
        model:      Loaded HybridMischiefModel (eval mode).
        frame:      BGR image array (H, W, 3).
        detections: YOLO detections with median_depth already filled.
        device:     Torch device string.

    Returns:
        "LOW", "MEDIUM", or "HIGH".
    """
    if not detections:
        return "LOW"

    img_t = preprocess_frame(frame, device)                       # (1, 3, 640, 640)
    feat_map, global_feat = backbone.extract(img_t)               # (1,256,40,40), (1,512)
    roi_feats = extract_roi_features(feat_map, detections)        # (N, 256)

    graph = build_static_graph(detections, roi_feats=roi_feats)
    graph_batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        logits = model(global_feat, graph_batch)                  # (1, 3)

    return LABEL_NAMES[int(logits.argmax(dim=1).item())]
