#!/usr/bin/env python3
"""
Task 3 — Fine-tune YOLOv11n on the curated dataset.

Training decisions (all justified in the report):
  model      : yolo11n.pt  — newest nano variant; pretrained on COCO, so
                             starting weights already know our 8 classes.
  imgsz      : 640         — YOLO standard; good detail-vs-compute trade-off.
  epochs     : 100         — sufficient for convergence; early stopping guards
                             against overfit.
  batch      : 16          — fits 8 GB GPU VRAM; adjust upward on better hw.
  optimizer  : AdamW       — more stable than SGD when fine-tuning pretrained.
  lr0        : 0.001       — conservative starting LR for fine-tuning.
  lrf        : 0.01        — cosine decay to lr0 × lrf = 1e-5 at epoch 100.
  patience   : 20          — stop if val mAP does not improve for 20 epochs.
  seed       : 42          — reproducibility.
  augmentation: Ultralytics mosaic + flip + HSV jitter defaults.

Outputs (model/runs/train/):
  weights/best.pt  — best checkpoint (by val mAP@0.5)
  weights/last.pt  — final epoch checkpoint
  results.csv      — per-epoch metrics
  results.png      — training curves

Run AFTER scripts/02_split_dataset.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

ROOT       = Path(__file__).parent.parent
SCHEMA_DIR = ROOT / "schema"
MODEL_DIR  = ROOT / "model"

# ---------------------------------------------------------------------------
# Hyperparameters — edit these to experiment
# ---------------------------------------------------------------------------
PRETRAINED_WEIGHTS = "yolo11n.pt"
DATASET_YAML       = str(SCHEMA_DIR / "dataset.yaml")
IMGSZ              = 640
EPOCHS             = 100
BATCH              = 16
OPTIMIZER          = "AdamW"
LR0                = 0.001
LRF                = 0.01
PATIENCE           = 20
SEED               = 42


def main() -> None:
    yaml_path = SCHEMA_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}\n"
            "Run scripts/02_split_dataset.py first."
        )

    print(f"Starting YOLOv11n fine-tuning on {DATASET_YAML}")
    model = YOLO(PRETRAINED_WEIGHTS)
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        patience=PATIENCE,
        seed=SEED,
        project=str(MODEL_DIR / "runs"),
        name="train",
        exist_ok=True,
        # Augmentation (explicitly set for reproducibility)
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    best = MODEL_DIR / "runs" / "train" / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")


if __name__ == "__main__":
    main()
