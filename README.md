# The Pet Mischief Detector

A pet mischief detection system using YOLOv11 object detection, Depth Anything V2 monocular depth estimation, and an optional Hybrid CNN+GNN risk classifier.

---

## Project Structure

```
data/
├── dataset/            YOLO training data (images + labels)
│   ├── train/
│   ├── val/
│   └── test/
└── clips/              Mischief clip labels for hybrid model training
    ├── clips.json
    └── images/

models/
├── gnn/                GNN temporal model modules
└── hybrid/             Hybrid CNN+GNN model modules
    └── checkpoints/    Saved hybrid model weights

model/
└── runs/train/weights/best.pt    YOLO fine-tuned weights

outputs/
└── visualizations/     Annotated output images

scripts/
├── 03_train.py         YOLO fine-tuning
├── 05_mischief_eval.py Heuristic mischief evaluation
├── 06_train_gnn.py     GNN temporal model training
└── 07_train_hybrid.py  Hybrid CNN+GNN training (precompute + train)
```

---

## Data Format

### YOLO Detection Training — `data/dataset/`

```
data/dataset/
├── train/
│   ├── images/    *.jpg / *.png
│   └── labels/    one *.txt per image
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/    (optional)
```

Each label file shares the same stem as its image. One object per line, space-separated, all values normalized to `[0, 1]`:

```
<class_id> <cx> <cy> <w> <h>
```

Example:
```
0 0.512 0.334 0.220 0.410
2 0.781 0.502 0.148 0.300
```

Class IDs:

| ID | Class |
|----|-------|
| 0  | cat   |
| 1  | dog   |
| 2  | chair |
| 3  | couch |
| 4  | bed   |
| 5  | dining table |
| 6  | toilet |
| 7  | person |

A `data.yaml` file must be present for YOLO training:

```yaml
path: data/dataset
train: train/images
val:   val/images
names:
  0: cat
  1: dog
  2: chair
  3: couch
  4: bed
  5: dining table
  6: toilet
  7: person
```

---

### Mischief Clip Labels — `data/clips/clips.json`

Used to train the Hybrid CNN+GNN model. Each entry is one clip (single image treated as a clip):

```json
[
  {
    "clip_id": "img_00",
    "frames": ["001"],
    "risk_level": "HIGH"
  },
  {
    "clip_id": "img_01",
    "frames": ["002"],
    "risk_level": "LOW"
  }
]
```

| Field | Description |
|-------|-------------|
| `clip_id` | Unique identifier. Cache file saved as `data/hybrid_cache/<clip_id>.pt` |
| `frames` | List of frame IDs. Each resolved to `data/clips/images/<frame_id>.{jpg,jpeg,png}` |
| `risk_level` | `"LOW"`, `"MEDIUM"`, or `"HIGH"` |

Place the corresponding images in `data/clips/images/`.

---

## Training Steps

### Step 1 — Train YOLO Detector

```bash
uv run scripts/03_train.py
```

Trains YOLOv11 on `data/dataset/`. Best weights are saved to:

```
model/runs/train/weights/best.pt
```

---

### Step 2 — Train Hybrid CNN+GNN Model

Requires `data/clips/clips.json` and images in `data/clips/images/`. Also requires YOLO weights from Step 1.

**Phase 1 — Extract and cache features** (YOLO + Depth Anything V2 + ResNet18 backbone):

```bash
uv run scripts/07_train_hybrid.py --phase precompute
```

Features are cached to `data/hybrid_cache/`.

**Phase 2 — Train HybridMischiefModel** on the cached features:

```bash
uv run scripts/07_train_hybrid.py --phase train
```

Best checkpoint saved to `models/hybrid/checkpoints/best.pt`.

Optional arguments:

```bash
uv run scripts/07_train_hybrid.py --phase train --epochs 80 --lr 5e-4 --batch-size 4 --val 0.2
```

---

## Evaluation Steps

### Heuristic Detector (default)

Runs YOLO + Depth Anything V2 + rule-based mischief scoring on all images in the input folder:

```bash
uv run main.py --input data/dataset/test/images --output outputs/visualizations
```

### Hybrid CNN+GNN Detector

Requires the trained hybrid checkpoint from Step 2:

```bash
uv run main.py --detector hybrid --input data/dataset/test/images --output outputs/hybrid
```

### Standalone Evaluation Script

Equivalent to the heuristic detector with a progress bar, useful for large test sets:

```bash
uv run scripts/05_mischief_eval.py
```

---

## All CLI Options

```
uv run main.py [OPTIONS]

--weights            Path to YOLO weights
                     (default: model/runs/train/weights/best.pt)
--input              Folder of input images
                     (default: data/dataset/test/images)
--output             Output folder for annotated images
                     (default: outputs/visualizations)
--detector           heuristic | hybrid  (default: heuristic)
--hybrid-checkpoint  Path to hybrid model checkpoint
                     (default: models/hybrid/checkpoints/best.pt)
```

---

## Key Paths

| What | Path |
|------|------|
| YOLO weights | `model/runs/train/weights/best.pt` |
| Clip labels | `data/clips/clips.json` |
| Clip images | `data/clips/images/` |
| Hybrid feature cache | `data/hybrid_cache/` |
| Hybrid checkpoint | `models/hybrid/checkpoints/best.pt` |
| Evaluation output | `outputs/visualizations/` |
