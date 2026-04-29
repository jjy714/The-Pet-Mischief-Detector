# Hybrid CNN + GNN — Analysis, Errors, and Advanced Implementation Plan

---

## Part 1 — Error and Weakness Analysis

### E1 — Missing `in_dim` after removing velocity features (concrete bug)

Section §8 says to remove `(vx, vy)` for static images, but the document never updates the node feature dimension. With velocity removed:

```
one_hot(8) + cx,cy,w,h(4) + depth(1) = 13 dims
```

The existing `graph_builder.py` produces `IN_DIM=15` (with velocity). The hybrid model needs its own `in_dim=13`. Using the wrong value causes a silent shape mismatch that crashes at the first `SAGEConv` forward pass.

---

### E2 — No edge features (architectural omission, not optional)

Section §4 says "No edge features (initially)" and §9 lists edge features as a "possible improvement." This is wrong framing. **Edge features are the primary signal for mischief detection**, not an enhancement.

The question this model must answer is: *how risky is the relationship between object A and object B?* That relationship is encoded entirely at the edge level:
- 2D distance
- depth difference (are they in the same plane?)
- relative direction (is the pet above/approaching the object?)
- pet-to-object flag

A GNN with no edge features can only use neighbor aggregation over node features. It learns "the cat has features X and the vase has features Y, therefore risk = Z" — it cannot learn "the cat is 5px from the vase with the same depth, therefore HIGH." The edge is what carries proximity and directionality.

GraphSAGE with no edge features reduces to: *average the neighbors' node embeddings*. For a fully connected graph this averages every object equally — defeating the purpose of the relational model entirely.

---

### E3 — CNN global feature is spatially detached from GNN nodes

The document proposes:
```
ResNet18 → global 512-pool → Linear(512, 128) → scene_feature
GNN → graph pooling → 128-dim relation_feature
concat → MLP → label
```

The problem: `scene_feature` is a single 128-dim vector encoding the entire image. The GNN receives no information about *where in the image* each object visually appears, only geometric coordinates. The CNN and GNN are fused only at the very end — they never communicate during processing.

The correct approach is **per-node visual enrichment**: extract a visual feature for each detected object from the CNN feature map using the object's bounding box (RoI crop), then add this as part of the node's input feature. This gives the GNN context like "this cat looks crouched" or "this vase is on a tilted surface."

Without it, the CNN branch contributes nothing that the GNN geometric features don't already capture (cx, cy, w, h, depth already encode position and scale).

---

### E4 — ResNet18 is too large; backbone is redundant with YOLO

ResNet18 has ~11M parameters. Fine-tuning it with 150 samples causes catastrophic overfitting. The document says "keep model small" but then proposes the largest component in the system.

More critically: the pipeline **already runs YOLO**, whose backbone (a CSPDarkNet-family network) produces rich intermediate feature maps. Using a separate ResNet18 means:
- Running two full backbone forward passes per image
- Two sets of backbone parameters in memory
- No sharing of visual representations

The backbone should be YOLO's own feature map, reused for the CNN branch at zero additional compute cost.

---

### E5 — No graph pooling specification

The document says the GNN outputs a "128-dim relational feature" but never states how node embeddings are aggregated to a graph-level vector. This is a concrete implementation gap.

GraphSAGE outputs one vector per node. For 5 objects that is a `(5, 128)` tensor. The document is silent on how this becomes a `(128,)` graph embedding. Options — mean pool, max pool, attention-weighted sum, a learned global node — have materially different behavior for mischief detection (max pool captures the "most alarmed" object; attention pool learns which pair drives risk).

---

### E6 — Batch size = 1, no PyG batching

PyG's `Batch.from_data_list()` handles variable-size graphs natively and enables `batch_size > 4`. With 150 samples and `batch_size=1`, each gradient step is based on a single sample — extremely noisy, slow convergence. This is fixable with a proper `collate_fn`.

---

### E7 — No dropout / regularization on the fusion head

With 150 samples and a 256-dim fusion MLP, overfitting is guaranteed without dropout. The document specifies no regularization at all on the classifier.

---

### E8 — Label space undefined

"Final label" is never specified. It must be `{LOW=0, MEDIUM=1, HIGH=2}` to align with the existing pipeline.

---

### Summary table

| # | Severity | Issue |
|---|---|---|
| E1 | Bug | Wrong `in_dim` if velocity removed |
| E2 | Architectural flaw | No edge features — core relational signal absent |
| E3 | Architectural flaw | CNN global feature doesn't inform GNN nodes |
| E4 | Design mistake | ResNet18 separate from YOLO — redundant + too large |
| E5 | Implementation gap | Graph pooling unspecified |
| E6 | Training flaw | batch_size=1, no PyG batching |
| E7 | Training flaw | No regularization on fusion head |
| E8 | Spec gap | Output label space undefined |

---

## Part 2 — Improved Architecture

### Core insight

The YOLO backbone is already running. Its intermediate feature map F (shape: `C×H'×W'`) is a free source of rich spatial features. Both the CNN branch and the GNN node features can be derived from it — eliminating the separate ResNet18 entirely.

### Revised architecture

```
Image
  │
  ▼
YOLO Backbone (frozen)
  ├── Feature map F  (C × H' × W')
  │     │
  │     ├── Global average pool → fc(C, 128) → scene_feat (128)
  │     │
  │     └── Per-node RoI crop (bilinear, fixed 7×7) → fc → visual_feat_i (64) per object
  │
  └── YOLO Head → detections
        │
        ▼
   Node features: [one_hot(8), cx,cy,w,h, depth, visual_feat(64)] = 77 dims per node
   Edge features: [dist_2d, depth_diff, rel_dx, rel_dy, is_pet_to_obj]  = 5 dims per edge
        │
        ▼
   GATConv layer 1  (in=77, out=64, heads=4, edge_dim=5) → 256 dims
   GATConv layer 2  (in=256, out=64, heads=1, edge_dim=5) → 64 dims
        │
        ▼
   Attention-weighted graph pool → relation_feat (64)
        │
        ▼
   concat(scene_feat(128), relation_feat(64)) → 192 dims
        │
        ▼
   MLP: Linear(192,64) → ReLU → Dropout(0.4) → Linear(64,3)
        │
        ▼
   LOW / MEDIUM / HIGH
```

### Why each change

| Change | Reason |
|---|---|
| Reuse YOLO backbone, remove ResNet18 | Eliminates 11M redundant parameters; saves compute; one forward pass |
| RoI crop → visual node feature | CNN and GNN now share spatial grounding; each node knows what the object visually looks like |
| Edge features (5-dim) | Encodes proximity and directionality — the actual mischief signal |
| GAT instead of GraphSAGE | Attention learns to weight nearby/risky pairs higher; weights are interpretable |
| Attention graph pool | Learns which node (the highest-risk object) should dominate the graph embedding |
| Dropout(0.4) on fusion | Required regularization for 150 samples |
| Explicit `batch_size > 1` with PyG Batch | Stable gradient estimates |

---

## Part 3 — Advanced Implementation Plan

### New files to create

```
models/
  hybrid/
    __init__.py
    graph_builder.py   # static variant: no vx,vy; adds edge features
    roi_pool.py        # RoI crop from feature map per detection
    model.py           # HybridMischiefModel
    dataset.py         # HybridDataset: yields (image_tensor, graph, label)
    train.py           # train_hybrid() function
    inference.py       # predict_image(model, frame, detections, depth_map)
scripts/
  07_train_hybrid.py
```

---

### Step 1 — `models/hybrid/graph_builder.py`

Static variant of the existing graph builder. Key differences:
- No `vx`, `vy` (no previous frame)
- `in_dim = 13` (8 one-hot + 4 bbox + 1 depth) before visual feature concat
- `in_dim = 77` after appending RoI visual feature (64-dim)
- **Edge features added** to the `Data` object as `edge_attr`

Edge feature vector for directed edge `i → j`:
```python
dist_2d      = euclidean_distance(centroid_i, centroid_j)          # 1
depth_diff   = depth_i - depth_j                                   # 1 (signed)
rel_dx       = cx_j - cx_i                                         # 1 (signed)
rel_dy       = cy_j - cy_i                                         # 1 (signed)
is_pet_to_obj = 1.0 if node_i is pet and node_j is not pet else 0  # 1
# total = 5 dims
```

Returns `Data(x=..., edge_index=..., edge_attr=...)`.

---

### Step 2 — `models/hybrid/roi_pool.py`

Extracts a fixed-size visual feature for each detection from a CNN feature map.

```python
def roi_pool_features(
    feature_map: torch.Tensor,   # (C, H', W') — YOLO backbone output
    detections: list[Detection],
    out_size: int = 7,
    proj_dim: int = 64,
) -> torch.Tensor:               # (N, proj_dim)
```

Implementation: for each bbox, compute the corresponding region in feature_map coordinates, use `torchvision.ops.roi_align` for differentiable bilinear RoI pooling (7×7), then adaptive-pool + flatten + linear projection to `proj_dim`.

This function is called once per image, producing one visual embedding per node.

---

### Step 3 — `models/hybrid/model.py`

```python
class HybridMischiefModel(nn.Module):
    def __init__(
        self,
        node_in_dim: int = 77,    # 13 geometric + 64 visual
        edge_dim:    int = 5,
        gnn_hidden:  int = 64,
        gnn_heads:   int = 4,
        scene_dim:   int = 128,
        num_classes: int = 3,
    ): ...
```

Components:
- `self.gat1 = GATConv(node_in_dim, gnn_hidden, heads=gnn_heads, edge_dim=edge_dim, concat=True)`
  → output: `gnn_hidden * gnn_heads = 256` per node
- `self.gat2 = GATConv(256, gnn_hidden, heads=1, edge_dim=edge_dim, concat=False)`
  → output: `gnn_hidden = 64` per node
- `self.attn_pool` — global attention pooling: learnable scoring MLP assigns each node a weight, weighted sum → graph embedding
- `self.scene_proj = nn.Linear(backbone_out_dim, scene_dim)` — projects global-pooled backbone features
- `self.classifier = nn.Sequential(Linear(scene_dim + gnn_hidden, 64), ReLU(), Dropout(0.4), Linear(64, num_classes))`

`forward(feature_map, graph)`:
1. Global pool `feature_map` → `scene_proj` → `scene_feat` (128)
2. GAT layers on `graph` → node embeddings (64 per node)
3. Attention pool → `relation_feat` (64)
4. Concat + classify → (1, 3)

Note: `feature_map` is provided externally (extracted from the YOLO backbone before the detection head). The model itself does not call YOLO.

---

### Step 4 — `models/hybrid/dataset.py`

`HybridDataset` — one sample per labeled image.

Precomputed and cached to disk (since running YOLO backbone on every training step is slow):

```python
class HybridSample(TypedDict):
    feature_map: torch.Tensor   # (C, H', W') — saved as .pt
    graph:       Data            # includes edge_attr
    label:       int

class HybridDataset(Dataset):
    def __init__(self, samples_dir: Path): ...
    # loads cached .pt files; yields (feature_map, graph, label)
```

**Preprocessing script** (called once before training):
- For each labeled image: run YOLO backbone (no head), save feature map + graph + label to `data/hybrid_cache/<clip_id>.pt`
- Then training reads only from cache — fast, no YOLO inference during training loop

---

### Step 5 — `models/hybrid/train.py`

```python
def collate_hybrid(batch):
    feature_maps, graphs, labels = zip(*batch)
    # stack feature maps (all same size after YOLO)
    fm_batch = torch.stack(feature_maps)
    # batch graphs with PyG
    g_batch  = Batch.from_data_list(graphs)
    label_t  = torch.tensor(labels, dtype=torch.long)
    return fm_batch, g_batch, label_t

def train_hybrid(
    dataset_dir: Path,
    checkpoint_path: str,
    epochs: int = 80,
    lr: float = 5e-4,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    device: str = "cpu",
) -> None: ...
```

Training notes:
- Backbone is fully frozen (no gradients flow into YOLO weights)
- `weight_decay=1e-4` on AdamW for regularization
- Early stopping on val loss with patience=15
- Use class-weighted `CrossEntropyLoss` if label distribution is skewed

---

### Step 6 — `models/hybrid/inference.py`

```python
def predict_image(
    model: HybridMischiefModel,
    yolo_backbone,               # callable: frame → feature_map tensor
    detections: list[Detection],
    device: str = "cpu",
) -> str:
    """Returns "LOW", "MEDIUM", or "HIGH" for a single frame."""
```

This is what replaces `calculate_mischief()` for the static/eval case.

---

### Step 7 — `scripts/07_train_hybrid.py`

Full training script:
1. Load YOLO backbone (freeze)
2. Precompute + cache feature maps and graphs for all labeled clips (if cache missing)
3. Build `HybridDataset` from cache
4. Train `HybridMischiefModel`
5. Save checkpoint to `models/hybrid/checkpoints/best.pt`

---

### Step 8 — Integration into `main.py`

Add `--detector hybrid` alongside the existing `heuristic` and `gnn` options.

In eval mode (`--detector hybrid`):
- Load `HybridMischiefModel` checkpoint
- Extract YOLO backbone feature map per image (one extra forward pass per image, not per clip)
- Call `predict_image()` → `risk_level`
- Run `calculate_mischief()` in parallel for pairs + warning_message (same Q2-a pattern as GNN)

In video mode: the hybrid model scores each frame independently (no sliding window needed — it's the per-frame scorer). The GNN+GRU temporal model remains the better choice for video mode.

---

## Part 4 — How the Three Detectors Relate

```
                   ┌─────────────────────────────────┐
   static image ──►│ Hybrid CNN+GNN                  │◄── eval mode
                   │ per-frame spatial + visual       │
                   └─────────────────────────────────┘

                   ┌─────────────────────────────────┐
   video stream ──►│ GNN + GRU (temporal)            │◄── video mode
                   │ clip-level spatial + temporal    │
                   └─────────────────────────────────┘

                   ┌─────────────────────────────────┐
     fallback  ──►│ Heuristic (calculate_mischief)  │◄── no trained model
                   │ pairwise rule-based scoring      │
                   └─────────────────────────────────┘
```

The three detectors are not competing — they serve different input modalities. `--detector hybrid` is the right choice when you have a single image and a trained hybrid model. `--detector gnn` is the right choice for real-time video. `--detector heuristic` runs without any trained weights.

---

## Part 5 — Feature Dimension Reference

| Feature | Dims | Present in temporal GNN | Present in hybrid CNN+GNN |
|---|---|---|---|
| one_hot(class) | 8 | yes | yes |
| cx, cy, w, h | 4 | yes | yes |
| depth | 1 | yes | yes |
| vx, vy (velocity) | 2 | yes | **no** (static images) |
| visual RoI feature | 64 | **no** | **yes** |
| **Node total** | **77 / 13** | **15** | **77 (13 + 64)** |
| edge_attr | 5 | **no** | **yes** |
