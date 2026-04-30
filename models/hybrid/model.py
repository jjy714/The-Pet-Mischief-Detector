from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GlobalAttention

from models.hybrid.backbone import GLOBAL_CHANNELS
from models.hybrid.graph_builder import EDGE_DIM, NODE_DIM
from models.hybrid.roi_pool import ROI_DIM

ROI_PROJ_DIM = 64
_NODE_IN_DIM = NODE_DIM + ROI_PROJ_DIM


# hybrid CNN + GNN mischief classifier that fuses a global scene feature with per-node graph attention
class HybridMischiefModel(nn.Module):

    GNN_HIDDEN  = 64
    GNN_HEADS   = 4
    SCENE_DIM   = 128
    NUM_CLASSES = 3

    def __init__(self) -> None:
        super().__init__()

        self.roi_proj = nn.Sequential(
            nn.Linear(ROI_DIM, ROI_PROJ_DIM),
            nn.ReLU(),
        )

        self.scene_proj = nn.Linear(GLOBAL_CHANNELS, self.SCENE_DIM)

        self.gat1 = GATConv(
            _NODE_IN_DIM,
            self.GNN_HIDDEN,
            heads=self.GNN_HEADS,
            edge_dim=EDGE_DIM,
            concat=True,
            add_self_loops=False,
        )

        self.gat2 = GATConv(
            self.GNN_HIDDEN * self.GNN_HEADS,
            self.GNN_HIDDEN,
            heads=1,
            edge_dim=EDGE_DIM,
            concat=False,
            add_self_loops=False,
        )

        self.attn_pool = GlobalAttention(gate_nn=nn.Linear(self.GNN_HIDDEN, 1))

        self.classifier = nn.Sequential(
            nn.Linear(self.SCENE_DIM + self.GNN_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.NUM_CLASSES),
        )

    ## fuse global scene feature with graph attention output and return (B, 3) logits
    def forward(self, global_feat: torch.Tensor, graph_batch: Batch) -> torch.Tensor:
        B = global_feat.shape[0]
        device = global_feat.device

        scene_feat = self.scene_proj(global_feat)

        if graph_batch.x.shape[0] == 0:
            relation_feat = torch.zeros(B, self.GNN_HIDDEN, device=device)
        else:
            geo   = graph_batch.x
            vis   = self.roi_proj(graph_batch.roi_feats)
            x     = torch.cat([geo, vis], dim=-1)

            edge_index = graph_batch.edge_index
            edge_attr  = graph_batch.edge_attr
            batch_vec  = graph_batch.batch

            x = self.gat1(x, edge_index, edge_attr).relu()
            x = self.gat2(x, edge_index, edge_attr)
            relation_feat = self.attn_pool(x, batch_vec)

        combined = torch.cat([scene_feat, relation_feat], dim=-1)
        return self.classifier(combined)
