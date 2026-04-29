from __future__ import annotations

import math

import torch
from torch_geometric.data import Data

from models.gnn.graph_builder import NUM_CLASSES, one_hot
from models.hybrid.roi_pool import ROI_DIM
from schema.Data import Detection

# Node features: [one_hot(8), cx, cy, w, h, depth] — no velocity for static images
NODE_DIM = NUM_CLASSES + 5   # 8 + 4 + 1 = 13
EDGE_DIM = 5                 # dist_2d, depth_diff, rel_dx, rel_dy, is_pet_to_obj

_PET_CLASS_IDS = {0, 1}      # cat=0, dog=1 (matches CLASS_NAMES in detector.py)
_DIAG = math.sqrt(2)         # diagonal of the unit square (normalized image coords)


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


def _centroid(det: Detection) -> tuple[float, float]:
    return (det.bbox.x_min + det.bbox.x_max) / 2, (det.bbox.y_min + det.bbox.y_max) / 2


def build_edges_and_attr(detections: list[Detection]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fully connected directed graph (no self-loops) with 5-dim edge features.

    Edge i → j encodes how object i relates to object j:
      dist_2d      : normalised 2D centroid distance ∈ [0, 1]
      depth_diff   : depth_i − depth_j  (signed; positive = i is closer)
      rel_dx       : cx_j − cx_i        (signed horizontal offset)
      rel_dy       : cy_j − cy_i        (signed vertical offset)
      is_pet_to_obj: 1 if i is a pet and j is a non-pet object, else 0
    """
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


def build_static_graph(
    detections: list[Detection],
    roi_feats: torch.Tensor | None = None,
) -> Data:
    """
    Build a static graph (no velocity) for the hybrid CNN+GNN model.

    Args:
        detections: YOLO detections with median_depth populated.
        roi_feats:  Optional (N, 256) RoI-pooled visual features per node.
                    Stored as graph.roi_feats for the model's roi_proj layer.

    Returns:
        torch_geometric.data.Data with:
          .x          (N, NODE_DIM=13)
          .edge_index (2, E)
          .edge_attr  (E, EDGE_DIM=5)
          .roi_feats  (N, 256)  — only if roi_feats is provided
    """
    n = len(detections)
    x = build_node_features(detections)
    edge_index, edge_attr = build_edges_and_attr(detections)
    if roi_feats is None:
        roi_feats = torch.zeros((n, ROI_DIM), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, roi_feats=roi_feats)
    return data
