from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from models.gnn.dataset import LABEL_MAP, LABEL_NAMES  # noqa: F401  (re-exported)


# dataset that loads pre-computed hybrid samples from a cache directory of .pt files
class HybridDataset(Dataset):

    def __init__(self, cache_dir: Path) -> None:
        self.files = sorted(Path(cache_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No cached hybrid samples found in {cache_dir}\n"
                "Run scripts/07_train_hybrid.py --precompute first."
            )

    ## return the number of cached samples
    def __len__(self) -> int:
        return len(self.files)

    ## load and return one (global_feat, graph, label) tuple
    def __getitem__(self, idx: int):
        sample = torch.load(self.files[idx], weights_only=False)
        return sample["global_feat"], sample["graph"], sample["label"]


## collate a list of (global_feat, graph, label) tuples into batched tensors
def collate_hybrid(batch: list) -> tuple[torch.Tensor, Batch, torch.Tensor]:
    global_feats, graphs, labels = zip(*batch)
    return (
        torch.stack(global_feats),
        Batch.from_data_list(list(graphs)),
        torch.tensor(labels, dtype=torch.long),
    )
