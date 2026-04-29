# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** `uv` — always prefix Python commands with `uv run`.

```bash
# Batch eval mode — annotated images saved under outputs/visualizations/{HIGH,MEDIUM,LOW}/
uv run main.py --mode eval
uv run main.py --mode eval --weights yolo11n.pt --input data/dataset/test/images --output outputs/visualizations

# Real-time video mode — press q to quit
uv run main.py --mode video --source 0        # webcam
uv run main.py --mode video --source video.mp4

# Risk label evaluation (produces outputs/risk_eval_<weights>.csv)
uv run scripts/06_eval_risk_labels.py --weights yolo26s.pt --images data/test/images --labels data/test/labels

# Data pipeline (run in order)
uv run scripts/01_curate_coco.py     # filter COCO 2017 → data/staged/
uv run scripts/02_split_dataset.py   # train/val/test split → data/dataset/
uv run scripts/03_train.py           # fine-tune YOLOv11n → model/runs/train/weights/best.pt
uv run scripts/04_evaluate.py        # YOLO mAP evaluation
uv run scripts/05_mischief_eval.py   # mischief scoring evaluation
uv run scripts/06_eval_risk_labels.py
```

## Architecture

The detector uses two models in tandem:
- **YOLO** (fine-tuned YOLOv11n or `yolo26s.pt`) — object detection for 8 classes: `cat`, `dog`, `cup`, `laptop`, `potted plant`, `vase`, `remote`, `keyboard`
- **Depth Anything V2 Small** (`depth-anything/Depth-Anything-V2-Small-hf`) — monocular depth estimation, normalized to a closeness map in [0, 1] (1 = near, 0 = far)

### Data flow

```
frame → infer_yolo() → [Detection, ...]
      → infer_depth() → depth_map (H×W float32)
                      → fill_depths()  # samples depth into each Detection.median_depth
                      → calculate_mischief()  # scores every (pet, object) pair
                      → draw_frame()   # annotates and returns BGR image
```

### Mischief scoring (`model/mischief.py`)

For every (pet, object) pair:
```
closeness = (W1 * proximity_2d + W3 * contact_likelihood) * depth_similarity
risk_score = closeness * pair_multiplier
```
- `proximity_2d` — edge-to-edge gap normalized by image diagonal (1.0 = touching, 0.0 = corners)
- `contact_likelihood` — IoU of bounding boxes
- `depth_similarity` — blended Depth Anything ordinal signal + size-based proxy; acts as a **multiplicative gate** (different Z-planes collapse the 2D score)
- `pair_multiplier` — per-pair table in `PAIR_MULTIPLIERS`; cat+cup/vase fire at lower closeness (×1.5)

Thresholds: `risk_score > 0.65` → HIGH, `> 0.3` → MEDIUM, else LOW. The single threshold + per-pair multipliers create effective per-pair thresholds (e.g. cat+cup fires at closeness ≥ 0.43).

### Video mode two-thread architecture (`main.py`)

- **Display thread**: reads every frame, runs YOLO (fast), reads the latest depth map from `_SharedState`, calls `calculate_mischief`, applies temporal hysteresis (15-frame `deque`, classifies by `min` of window — requires ~1 s sustained contact before HIGH fires), renders and shows frame.
- **Depth worker thread**: continuously reads the latest frame from `_SharedState`, runs Depth Anything V2, writes EMA-normalized depth back. Alpha = 0.1 gives ~1 s time constant at 10 FPS depth updates, stabilizing the closeness scale across scene changes.
- Eval mode (`run_eval`) uses per-frame normalization via `infer_depth()` — no EMA, no hysteresis.

### Key files

| File | Purpose |
|------|---------|
| `main.py` | Entry point; `run_eval` / `run_video` / `_SharedState` / `_depth_worker` |
| `model/detector.py` | `load_yolo`, `load_depth_model`, `infer_yolo`, `infer_depth`, `_run_depth_model`, `fill_depths` |
| `model/mischief.py` | `calculate_mischief`, `_classify`, scoring helpers, `PAIR_MULTIPLIERS` |
| `model/visualize.py` | `draw_frame` — bounding boxes, risk banner, depth thumbnail, FPS overlay |
| `schema/Data.py` | Pydantic models: `BoundingBox`, `Detection`, `PairRisk`, `MischiefResult` |
| `schema/dataset.yaml` | YOLO dataset config pointing at `data/dataset/` |
| `scripts/01–06_*.py` | Numbered data-pipeline scripts; run in order |

### Data layout

```
data/raw/          ← COCO 2017 source (not committed)
data/staged/       ← filtered by script 01
data/dataset/      ← train/val/test split by script 02; schema/dataset.yaml points here
data/test/         ← hand-labelled risk-level test set (labels: 0=LOW, 1=MEDIUM, 2=HIGH)
```

YOLO weights: `yolo11n.pt` and `yolo26s.pt` in repo root; fine-tuned weights at `model/runs/train/weights/best.pt`.

### Design notes from `docs/`

- Depth is a **multiplicative gate**, not an additive term — prevents false positives from 2D screen geometry when objects are at different depths.
- `fill_depths` samples a narrow portrait strip (center 20% width, dropping top 12.5% height) at the 25th percentile to bias toward each object's near surface.
- `_size_depth_proxy` blends a size-based metric depth estimate into `depth_similarity`; blend weight scales by `min_m / max_m` so high-variance classes (dog) fall back to raw Depth Anything automatically.
