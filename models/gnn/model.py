from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

IN_DIM = 15  # 8 one-hot + cx,cy,w,h + depth + vx,vy


class MischiefGNN(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # LOW, MEDIUM, HIGH
        )

    def _embed_graph(self, graph: Data) -> torch.Tensor:
        x, edge_index = graph.x, graph.edge_index
        if x.shape[0] == 0:
            device = next(self.parameters()).device
            return torch.zeros(self.hidden, device=device)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)  # (hidden,)

    def forward(self, graph_sequence: list[Data]) -> torch.Tensor:
        embeddings = [self._embed_graph(g) for g in graph_sequence]
        seq = torch.stack(embeddings).unsqueeze(0)  # (1, T, hidden)
        out, _ = self.gru(seq)
        return self.classifier(out[:, -1, :])  # (1, 3)
