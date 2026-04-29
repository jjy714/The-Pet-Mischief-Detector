from __future__ import annotations

import torch

from models.gnn.dataset import LABEL_NAMES
from models.gnn.graph_builder import build_graph
from models.gnn.model import IN_DIM, MischiefGNN
from schema.Data import Detection


def load_gnn_model(checkpoint_path: str, device: str = "cpu") -> MischiefGNN:
    model = MischiefGNN(in_dim=IN_DIM).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_clip(
    model: MischiefGNN,
    frame_detections: list[list[Detection]],
    device: str = "cpu",
) -> str:
    """
    Args:
        model:             Loaded MischiefGNN in eval mode.
        frame_detections:  Per-frame Detection lists with median_depth filled.
        device:            Torch device string.

    Returns:
        "LOW", "MEDIUM", or "HIGH".
    """
    graphs = []
    prev: list[Detection] | None = None
    for detections in frame_detections:
        graphs.append(build_graph(detections, prev).to(device))
        prev = detections

    with torch.no_grad():
        logits = model(graphs)

    return LABEL_NAMES[int(logits.argmax(dim=1).item())]
