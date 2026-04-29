from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from models.gnn.dataset import LABEL_MAP, LABEL_NAMES  # noqa: F401  (re-exported)


class HybridDataset(Dataset):
    """
    Loads pre-computed hybrid samples from a cache directory.

    Each cache file is a dict saved with torch.save:
      {
        "global_feat": Tensor (512,),
        "graph":       torch_geometric.data.Data
                         .x          (N, 13)   geometric node features
                         .edge_index (2, E)
                         .edge_attr  (E, 5)
                         .roi_feats  (N, 256)  RoI-pooled visual features
        "label":       int  (0=LOW, 1=MEDIUM, 2=HIGH)
      }

    Files are expected at: <cache_dir>/<clip_id>.pt
    """

    def __init__(self, cache_dir: Path) -> None:
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached hybrid samples found in {cache_dir}\n"
                "Run scripts/07_train_hybrid.py --precompute first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        sample = torch.load(self.files[idx], weights_only=False)
        return sample["global_feat"], sample["graph"], sample["label"]


def collate_hybrid(batch: list) -> tuple[torch.Tensor, Batch, torch.Tensor]:
    """
    DataLoader collate for (global_feat, graph, label) tuples.

    Stacks global_feat tensors and uses PyG Batch to merge variable-size graphs.
    """
    global_feats, graphs, labels = zip(*batch)
    return (
        torch.stack(global_feats),          # (B, 512)
        Batch.from_data_list(list(graphs)), # PyG Batch
        torch.tensor(labels, dtype=torch.long),
    )
