#!/usr/bin/env python3
"""
Train the GNN mischief detector on clip-labeled data.

Clips file format (JSON array):
  [
    {
      "clip_id":    "img_00",
      "frames":     ["001"],
      "risk_level": "HIGH",
      "reason":     "cat near vase"
    },
    {
      "clip_id":    "video1_clip_03",
      "frames":     ["00020", "00021", "00022", "00023"],
      "risk_level": "HIGH",
      "reason":     "cat approaching vase quickly"
    }
  ]

Each frame ID is resolved to:
  <images_dir>/<frame_id>.jpg  (falls back to .jpeg then .png)

For single-image clips, velocity features (vx, vy) are zero — the
GraphSAGE spatial module still trains normally; GRU temporal
dynamics require multi-frame clips.

Run:
  uv run scripts/06_train_gnn.py
  uv run scripts/06_train_gnn.py --clips data/clips/clips.json \\
      --images data/clips/images --epochs 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch

from models.detector import fill_depths, infer_depth, infer_yolo, load_depth_model, load_yolo
from models.gnn.train import train_gnn
from schema.Data import Detection

ROOT       = Path(__file__).parent.parent
WEIGHTS    = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
CLIPS_JSON = ROOT / "data" / "clips" / "clips.json"
IMAGES_DIR = ROOT / "data" / "clips" / "images"
CKPT_DIR   = ROOT / "models" / "gnn" / "checkpoints"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_frame_loader(yolo, processor, depth_model, device: str, images_dir: Path):
    """
    Returns a frame_loader(frame_id) -> list[Detection] closure.

    Resolves frame_id to an image file, runs YOLO + Depth Anything V2,
    and returns detections with median_depth populated.
    Returns an empty list if the image cannot be found or read.
    """
    def frame_loader(frame_id: str) -> list[Detection]:
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = images_dir / f"{frame_id}{ext}"
            if img_path.exists():
                break
        else:
            print(f"  WARNING: frame not found: {images_dir / frame_id}.*")
            return []

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path}")
            return []

        h, w       = frame.shape[:2]
        detections = infer_yolo(yolo, frame)
        depth_map  = infer_depth(processor, depth_model, frame, device)
        return fill_depths(detections, depth_map, h, w)

    return frame_loader


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MischiefGNN")
    parser.add_argument("--clips",   default=str(CLIPS_JSON), help="Path to clips JSON file")
    parser.add_argument("--images",  default=str(IMAGES_DIR), help="Directory containing frame images")
    parser.add_argument("--weights", default=str(WEIGHTS),    help="YOLO weights path")
    parser.add_argument("--epochs",  default=50,  type=int)
    parser.add_argument("--lr",      default=1e-3, type=float)
    parser.add_argument("--val",     default=0.2,  type=float, help="Validation fraction (default: 0.2)")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    clips_path = Path(args.clips)
    if not clips_path.exists():
        raise FileNotFoundError(
            f"Clips file not found: {clips_path}\n"
            "Create data/clips/clips.json with clip-level labels first."
        )
    clips = json.loads(clips_path.read_text())
    print(f"Loaded {len(clips)} clips from {clips_path}")

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}\n"
            "Run scripts/03_train.py first."
        )

    print("Loading YOLO ...")
    yolo = load_yolo(str(weights))
    print("Loading Depth Anything V2 ...")
    processor, depth_model = load_depth_model(device)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(CKPT_DIR / "best.pt")

    frame_loader = make_frame_loader(
        yolo, processor, depth_model, device, Path(args.images)
    )

    print(f"Training for {args.epochs} epochs ...")
    train_gnn(
        clips=clips,
        frame_loader=frame_loader,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        lr=args.lr,
        val_fraction=args.val,
        device=device,
    )


if __name__ == "__main__":
    main()
