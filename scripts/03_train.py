#!/usr/bin/env python3
"""
Task 3 — Fine-tune a YOLO model on the curated dataset.

Supports all experiment configurations from docs/finetuning_plan.md via CLI args.
Defaults reproduce the original baseline (yolo11n, imgsz=640).

Usage:
  # Baseline
  uv run scripts/03_train.py

  # Exp A — higher resolution
  uv run scripts/03_train.py --imgsz 1280 --batch 4 --name exp_a

  # Exp B — copy-paste + augmentation
  uv run scripts/03_train.py --copy-paste 0.5 --mixup 0.15 --erasing 0.4 --scale 0.9 --name exp_b

  # Exp C — model upgrade
  uv run scripts/03_train.py --weights yolo26s.pt --batch 8 --lr0 0.0005 --warmup-epochs 5 --patience 25 --copy-paste 0.3 --mixup 0.1 --name exp_c

Outputs (model/runs/<name>/):
  weights/best.pt  — best checkpoint (by val mAP@0.5)
  weights/last.pt  — final epoch checkpoint
  results.csv      — per-epoch metrics
  results.png      — training curves

Run AFTER scripts/02_split_dataset.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO


def _default_device() -> str:
    if torch.cuda.is_available():
        return "0"          # first CUDA GPU
    if torch.backends.mps.is_available():
        return "mps"        # Apple Silicon GPU
    return "cpu"

ROOT       = Path(__file__).parent.parent
SCHEMA_DIR = ROOT / "schema"
MODEL_DIR  = ROOT / "model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune YOLO on the pet-mischief dataset.")
    p.add_argument("--weights",        default="yolo11n.pt",  help="Pretrained backbone weights")
    p.add_argument("--imgsz",          type=int,   default=640,   help="Training image size")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch",          type=int,   default=16)
    p.add_argument("--lr0",            type=float, default=0.001, help="Initial learning rate")
    p.add_argument("--lrf",            type=float, default=0.01,  help="Final LR = lr0 * lrf")
    p.add_argument("--warmup-epochs",  type=int,   default=3,     dest="warmup_epochs")
    p.add_argument("--patience",       type=int,   default=20,    help="Early-stop patience (epochs)")
    p.add_argument("--copy-paste",     type=float, default=0.0,   dest="copy_paste")
    p.add_argument("--mixup",          type=float, default=0.0)
    p.add_argument("--erasing",        type=float, default=0.4,   help="Random erasing probability")
    p.add_argument("--scale",          type=float, default=0.5,   help="Scale jitter range")
    p.add_argument("--device",         default=_default_device(), help="Device: 0 (CUDA), mps, cpu")
    p.add_argument("--name",           default="train",           help="Run name under model/runs/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yaml_path = SCHEMA_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}\n"
            "Run scripts/02_split_dataset.py first."
        )

    print(f"Starting fine-tuning: {args.weights}  imgsz={args.imgsz}  batch={args.batch}  device={args.device}  name={args.name}")
    model = YOLO(args.weights)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=42,
        project=str(MODEL_DIR / "runs"),
        name=args.name,
        exist_ok=True,
        # Augmentation
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        copy_paste=args.copy_paste,
        mixup=args.mixup,
        erasing=args.erasing,
        scale=args.scale,
    )

    best = MODEL_DIR / "runs" / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")


if __name__ == "__main__":
    main()
