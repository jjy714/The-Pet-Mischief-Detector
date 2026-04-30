#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO


## pick the best available device: first CUDA GPU, then Apple Silicon, then CPU
def _default_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

ROOT       = Path(__file__).parent.parent
SCHEMA_DIR = ROOT / "schema"
MODEL_DIR  = ROOT / "model"


## parse CLI args for all fine-tuning hyperparameters
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
    p.add_argument("--fraction",        type=float, default=1.0,   help="Fraction of training data to use (0.0–1.0)")
    p.add_argument("--freeze",          type=int,   default=0,     help="Freeze first N backbone layers (0 = none)")
    p.add_argument("--name",           default="train",           help="Run name under model/runs/")
    return p.parse_args()


## load YOLO, run training, and print the best checkpoint path
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
        fraction=args.fraction,
        freeze=args.freeze if args.freeze > 0 else None,
        seed=42,
        project=str(MODEL_DIR / "runs"),
        name=args.name,
        exist_ok=True,
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
