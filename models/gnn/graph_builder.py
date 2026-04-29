from __future__ import annotations

import torch
from torch_geometric.data import Data

from schema.Data import Detection

NUM_CLASSES = 8  # must match CLASS_NAMES in models/detector.py
# Feature vector layout: [one_hot(8), cx, cy, w, h, depth, vx, vy] = 15 dims


def one_hot(class_id: int, num_classes: int = NUM_CLASSES) -> list[float]:
    vec = [0.0] * num_classes
    if 0 <= class_id < num_classes:
        vec[class_id] = 1.0
    return vec


def find_match(det: Detection, prev_detections: list[Detection]) -> Detection | None:
    """Closest same-class detection in prev_detections by centroid distance."""
    cx = (det.bbox.x_min + det.bbox.x_max) / 2
    cy = (det.bbox.y_min + det.bbox.y_max) / 2
    best, best_dist = None, float("inf")
    for prev in prev_detections:
        if prev.class_id != det.class_id:
            continue
        px = (prev.bbox.x_min + prev.bbox.x_max) / 2
        py = (prev.bbox.y_min + prev.bbox.y_max) / 2
        d = (cx - px) ** 2 + (cy - py) ** 2
        if d < best_dist:
            best, best_dist = prev, d
    return best


def build_node_features(
    detections: list[Detection],
    prev_detections: list[Detection] | None = None,
) -> torch.Tensor:
    if not detections:
        return torch.zeros((0, NUM_CLASSES + 7), dtype=torch.float)

    feats = []
    for det in detections:
        cx = (det.bbox.x_min + det.bbox.x_max) / 2
        cy = (det.bbox.y_min + det.bbox.y_max) / 2
        w  = det.bbox.x_max - det.bbox.x_min
        h  = det.bbox.y_max - det.bbox.y_min

        vx, vy = 0.0, 0.0
        if prev_detections:
            match = find_match(det, prev_detections)
            if match:
                prev_cx = (match.bbox.x_min + match.bbox.x_max) / 2
                prev_cy = (match.bbox.y_min + match.bbox.y_max) / 2
                vx = cx - prev_cx
                vy = cy - prev_cy

        feat = [
            *one_hot(det.class_id),
            cx, cy, w, h,
            det.median_depth,
            vx, vy,
        ]
        feats.append(feat)

    return torch.tensor(feats, dtype=torch.float)


def build_edges(num_nodes: int) -> torch.Tensor:
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = [
        [i, j]
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    ]
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def build_graph(
    detections: list[Detection],
    prev_detections: list[Detection] | None = None,
) -> Data:
    x = build_node_features(detections, prev_detections)
    edge_index = build_edges(len(detections))
    return Data(x=x, edge_index=edge_index)
