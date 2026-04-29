# GNN + Temporal Mischief Detection — Summary

## 1. Objective
Upgrade the existing proximity-based mischief detector into a **spatio-temporal reasoning system** using a Graph Neural Network (GNN) with temporal modeling.

---

## 2. Key Insight
- Current system = proximity detector (static, frame-based)
- Target system = behavior-aware detector (temporal, relational)

Mischief is **not spatial only** — it is **temporal interaction**.

---

## 3. Final Architecture

```
Frame → YOLO → Depth → Graph (nodes + edges)
       → Graph Sequence (clip)
       → GNN (GraphSAGE)
       → GRU (temporal modeling)
       → Risk Classification (LOW / MEDIUM / HIGH)
```

---

## 4. Data Format (Clip-based Labels)

```
{
  "clip_id": "video1_clip_03",
  "frames": ["00020", "00021", "00022", "00023"],
  "risk_level": "HIGH",
  "reason": "cat approaching vase quickly"
}
```

- Each sample = short sequence of frames
- Label reflects **temporal behavior**, not static proximity

---

## 5. Graph Construction

### Nodes (per object)
Features:
- class (one-hot)
- bbox (cx, cy, w, h)
- depth
- velocity (from previous frame)


### Edges (between all objects)
- fully connected graph
- captures object interactions

---

## 6. Model Design

### Spatial Module (GraphSAGE)
- 2 layers
- learns object relationships

### Temporal Module (GRU)
- processes sequence of graphs
- captures motion and interaction over time

### Output
- 3-class classification (LOW / MEDIUM / HIGH)

---

## 7. Training Strategy

- Input: sequence of graphs (clip)
- Label: clip-level risk
- Loss: CrossEntropyLoss

### Data Constraints (~150 samples)
- keep model small (hidden dim ≤ 64)
- avoid overfitting

Optional:
- use heuristic system as pseudo-label for augmentation

---

## 8. Integration Steps

1. Build graph per frame from detections
2. Stack graphs into clip sequences
3. Train GNN + GRU model
4. Replace `calculate_mischief` with GNN inference
5. Use sliding window for real-time video

---

## 9. Improvements Over Original System

| Aspect | Original | New |
|------|--------|-----|
| Reasoning | Pairwise | Multi-object graph |
| Temporal | None | GRU-based |
| Multipliers | Manual | Learned |
| False positives | High (static) | Reduced (temporal filtering) |

---

## 10. Limitations

- Still lacks fine-grained intent (no pose/action modeling)
- Small dataset limits generalization
- Requires good detection + depth quality

---

## 11. Final Recommendation

- Use **GraphSAGE + GRU** (not vanilla GCN)
- Train on **clip-level labels**
- Keep existing system as fallback / pseudo-label source

---

## 12. Final Takeaway

This upgrade transforms the system from:

> "Are objects close?"

into:

> "Is the pet interacting with the object over time in a risky way?"

which is the correct formulation of the problem.
