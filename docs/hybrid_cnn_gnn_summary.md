# Hybrid CNN + GNN Model — Summary

## 1. Objective
Build a model that understands **semantic relationships in static images** (e.g., "cat hitting vase", "dog beside remote") by combining:
- Global scene understanding (CNN)
- Object-level relational reasoning (GNN)

---

## 2. Final Architecture

```
Image
 ├── CNN → global feature
 └── YOLO → detections → Graph → GNN → relation feature
                         ↓
               concatenate (CNN + GNN)
                         ↓
                    classifier
```

---

## 3. Components

### CNN Encoder
- Backbone: ResNet18 (pretrained)
- Output: 128-dim global feature
- Role: captures scene context, textures, and pose hints

### GNN Encoder
- Model: GraphSAGE (2 layers)
- Input: object graph from detections
- Output: 128-dim relational feature
- Role: captures object interactions and spatial relationships

### Fusion + Classifier
- Concatenate CNN + GNN features → 256-dim
- MLP classifier → final label

---

## 4. Graph Design

### Nodes (per object)
Features:
- one-hot class
- bounding box (cx, cy, w, h)
- depth

### Edges
- Fully connected graph
- No edge features (initially)

---

## 5. Data Pipeline

Each sample contains:
```
image → tensor
YOLO → detections → graph
label → class index
```

Important:
- Precompute YOLO + depth outputs
- Do NOT run detection during training

---

## 6. Training Setup

- Loss: CrossEntropyLoss
- Optimizer: Adam
- Batch size: 1 (initially)

Training loop:
```
pred = model(image, graph)
loss = criterion(pred, label)
```

---

## 7. Key Advantages

### CNN
- Captures global scene information
- Learns visual cues (pose, texture)

### GNN
- Captures object relationships
- Learns spatial interactions

### Combined
- Enables semantic reasoning:
  > "A cat near a vase in a risky pose → likely hitting"

---

## 8. Important Notes

- Remove temporal features (vx, vy)
- Keep model small due to limited dataset (~150 samples)
- Use fully connected graph initially for simplicity

---

## 9. Possible Improvements

- Add edge features (distance, depth difference)
- Replace GraphSAGE with GAT (attention-based)
- Improve fusion (attention instead of concatenation)

---

## 10. Final Takeaway

The hybrid CNN + GNN model combines **pixel-level understanding** with **object-level reasoning**, making it suitable for detecting semantic relationships in static images.

