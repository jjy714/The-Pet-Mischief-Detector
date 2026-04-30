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


## convert a BGR OpenCV frame to a (1, 3, 640, 640) ImageNet-normalised tensor
def preprocess_frame(frame: np.ndarray, device: str = "cpu") -> torch.Tensor:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t   = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    t   = TF.resize(t, [_IMG_SIZE, _IMG_SIZE], antialias=True)
    t   = TF.normalize(t, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return t.unsqueeze(0).to(device)


## load the frozen backbone and trained HybridMischiefModel from a checkpoint file
def load_hybrid_model(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[ResNetBackbone, HybridMischiefModel]:
    backbone = ResNetBackbone().to(device)
    backbone.eval()

    model = HybridMischiefModel().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return backbone, model


## predict mischief risk level for a single static frame given YOLO detections with depth filled
def predict_image(
    backbone: ResNetBackbone,
    model: HybridMischiefModel,
    frame: np.ndarray,
    detections: list[Detection],
    device: str = "cpu",
) -> str:
    if not detections:
        return "LOW"

    img_t = preprocess_frame(frame, device)
    feat_map, global_feat = backbone.extract(img_t)
    roi_feats = extract_roi_features(feat_map, detections)

    graph = build_static_graph(detections, roi_feats=roi_feats)
    graph_batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        logits = model(global_feat, graph_batch)

    return LABEL_NAMES[int(logits.argmax(dim=1).item())]
