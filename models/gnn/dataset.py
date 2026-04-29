from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

from models.gnn.graph_builder import build_graph
from schema.Data import Detection

LABEL_MAP: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Callable contract: accepts a frame_id string, returns list[Detection] with
# median_depth already populated (YOLO + depth inference done by the caller).
FrameLoader = Callable[[str], list[Detection]]


class ClipDataset(Dataset):
    def __init__(self, clips: list[dict], frame_loader: FrameLoader):
        self.clips = clips
        self.frame_loader = frame_loader

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[list, torch.Tensor]:
        clip = self.clips[idx]
        graphs = []
        prev_det: list[Detection] | None = None
        for frame_id in clip["frames"]:
            detections = self.frame_loader(frame_id)
            graphs.append(build_graph(detections, prev_det))
            prev_det = detections
        label = torch.tensor(LABEL_MAP[clip["risk_level"]], dtype=torch.long)
        return graphs, label
