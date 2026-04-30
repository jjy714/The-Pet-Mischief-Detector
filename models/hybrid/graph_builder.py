from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from models.gnn.graph_builder import NUM_CLASSES, one_hot
from models.hybrid.roi_pool import ROI_DIM
from schema.Data import Detection

### NODE_DIM layout: [one_hot(8 classes), cx, cy, w, h, depth] = 13 dims (no velocity; static image)
NODE_DIM = NUM_CLASSES + 5
### EDGE_DIM layout: [dist_2d, depth_diff, rel_dx, rel_dy, is_pet_to_obj] = 5 dims
EDGE_DIM = 5

_PET_CLASS_IDS = {0, 1}
_DIAG = math.sqrt(2)


## build a (N, NODE_DIM) node feature tensor from detection list
def build_node_features(detections: list[Detection]) -> torch.Tensor:
    if not detections:
        return torch.zeros((0, NODE_DIM), dtype=torch.float)
    feats = []
    for det in detections:
        cx = (det.bbox.x_min + det.bbox.x_max) / 2
        cy = (det.bbox.y_min + det.bbox.y_max) / 2
        w  = det.bbox.x_max - det.bbox.x_min
        h  = det.bbox.y_max - det.bbox.y_min
        feats.append([*one_hot(det.class_id), cx, cy, w, h, det.median_depth])
    return torch.tensor(feats, dtype=torch.float)


## return the normalized centroid of a detection bounding box
def _centroid(det: Detection) -> tuple[float, float]:
    return (det.bbox.x_min + det.bbox.x_max) / 2, (det.bbox.y_min + det.bbox.y_max) / 2


## build fully connected directed edge_index and edge_attr tensors for all detection pairs
def build_edges_and_attr(detections: list[Detection]) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(detections)
    if n < 2:
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, EDGE_DIM), dtype=torch.float),
        )

    centroids = [_centroid(d) for d in detections]
    edge_idx, edge_feats = [], []

    for i, di in enumerate(detections):
        cxi, cyi = centroids[i]
        for j, dj in enumerate(detections):
            if i == j:
                continue
            cxj, cyj = centroids[j]
            dist_2d      = math.sqrt((cxj - cxi) ** 2 + (cyj - cyi) ** 2) / _DIAG
            depth_diff   = di.median_depth - dj.median_depth
            rel_dx       = cxj - cxi
            rel_dy       = cyj - cyi
            is_pet_to_obj = 1.0 if (
                di.class_id in _PET_CLASS_IDS and dj.class_id not in _PET_CLASS_IDS
            ) else 0.0
            edge_idx.append([i, j])
            edge_feats.append([dist_2d, depth_diff, rel_dx, rel_dy, is_pet_to_obj])

    edge_index = torch.tensor(edge_idx,   dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(edge_feats, dtype=torch.float)
    return edge_index, edge_attr


## build a static PyG Data graph for the hybrid CNN+GNN model, optionally attaching RoI visual features
def build_static_graph(
    detections: list[Detection],
    roi_feats: torch.Tensor | None = None,
) -> Data:
    n = len(detections)
    x = build_node_features(detections)
    edge_index, edge_attr = build_edges_and_attr(detections)
    if roi_feats is None:
        roi_feats = torch.zeros((n, ROI_DIM), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, roi_feats=roi_feats)
    return data
