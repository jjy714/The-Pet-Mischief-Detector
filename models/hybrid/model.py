from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GlobalAttention

from models.hybrid.backbone import GLOBAL_CHANNELS
from models.hybrid.graph_builder import EDGE_DIM, NODE_DIM
from models.hybrid.roi_pool import ROI_DIM

# Projected visual feature dimension (trainable linear inside the model)
ROI_PROJ_DIM = 64
# After appending projected visual feat to geometric feat: 13 + 64 = 77
_NODE_IN_DIM = NODE_DIM + ROI_PROJ_DIM


class HybridMischiefModel(nn.Module):
    """
    Hybrid CNN + GNN mischief classifier for static images.

    Forward inputs:
      global_feat  (B, 512)  — ResNet18 avgpool output (frozen backbone)
      graph_batch  Batch     — PyG batched graphs, must carry:
                               .x          (N_total, 13)   geometric node features
                               .roi_feats  (N_total, 256)  raw RoI visual features
                               .edge_index (2, E_total)
                               .edge_attr  (E_total, 5)
                               .batch      (N_total,)      node-to-graph mapping

    Forward output:
      (B, 3) logits for [LOW, MEDIUM, HIGH]

    Architecture:
      roi_proj  : Linear(256, 64)  — projects RoI features; trainable
      scene_proj: Linear(512, 128) — projects global scene feature; trainable
      gat1      : GATConv(77→64×4=256, edge_dim=5, add_self_loops=False)
      gat2      : GATConv(256→64,      edge_dim=5, add_self_loops=False)
      attn_pool : GlobalAttention(Linear(64,1)) → (B, 64)
      classifier: Linear(192,64) → ReLU → Dropout(0.4) → Linear(64,3)
    """

    GNN_HIDDEN  = 64
    GNN_HEADS   = 4
    SCENE_DIM   = 128
    NUM_CLASSES = 3

    def __init__(self) -> None:
        super().__init__()

        # Visual feature projection (trainable — adapts RoI features to task)
        self.roi_proj = nn.Sequential(
            nn.Linear(ROI_DIM, ROI_PROJ_DIM),
            nn.ReLU(),
        )

        # Scene feature projection
        self.scene_proj = nn.Linear(GLOBAL_CHANNELS, self.SCENE_DIM)

        # GAT layers — edge features carry proximity and directionality
        self.gat1 = GATConv(
            _NODE_IN_DIM,
            self.GNN_HIDDEN,
            heads=self.GNN_HEADS,
            edge_dim=EDGE_DIM,
            concat=True,
            add_self_loops=False,
        )  # output: GNN_HIDDEN * GNN_HEADS = 256 per node

        self.gat2 = GATConv(
            self.GNN_HIDDEN * self.GNN_HEADS,
            self.GNN_HIDDEN,
            heads=1,
            edge_dim=EDGE_DIM,
            concat=False,
            add_self_loops=False,
        )  # output: GNN_HIDDEN = 64 per node

        # Attention-weighted graph pooling — learns which node drives risk
        self.attn_pool = GlobalAttention(gate_nn=nn.Linear(self.GNN_HIDDEN, 1))

        # Fusion classifier with dropout for small-dataset regularisation
        self.classifier = nn.Sequential(
            nn.Linear(self.SCENE_DIM + self.GNN_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.NUM_CLASSES),
        )

    def forward(self, global_feat: torch.Tensor, graph_batch: Batch) -> torch.Tensor:
        """
        Args:
            global_feat: (B, 512)
            graph_batch: PyG Batch with .x, .roi_feats, .edge_index,
                         .edge_attr, .batch

        Returns:
            (B, 3) logits.
        """
        B = global_feat.shape[0]
        device = global_feat.device

        # ── Scene feature ────────────────────────────────────────────────────
        scene_feat = self.scene_proj(global_feat)           # (B, 128)

        # ── GNN branch ───────────────────────────────────────────────────────
        if graph_batch.x.shape[0] == 0:
            # All graphs in this batch have no detections
            relation_feat = torch.zeros(B, self.GNN_HIDDEN, device=device)
        else:
            geo   = graph_batch.x                               # (N_total, 13)
            vis   = self.roi_proj(graph_batch.roi_feats)        # (N_total, 64)
            x     = torch.cat([geo, vis], dim=-1)               # (N_total, 77)

            edge_index = graph_batch.edge_index
            edge_attr  = graph_batch.edge_attr
            batch_vec  = graph_batch.batch                      # (N_total,)

            x = self.gat1(x, edge_index, edge_attr).relu()     # (N_total, 256)
            x = self.gat2(x, edge_index, edge_attr)             # (N_total, 64)
            relation_feat = self.attn_pool(x, batch_vec)        # (B, 64)

        # ── Fusion + classify ─────────────────────────────────────────────────
        combined = torch.cat([scene_feat, relation_feat], dim=-1)  # (B, 192)
        return self.classifier(combined)                            # (B, 3)
