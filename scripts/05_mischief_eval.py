#!/usr/bin/env python3
"""
Task 4B — Qualitative mischief system evaluation (batch image mode).

Runs the full pipeline on every image in the test set:
  1. YOLO detection (bounding boxes + class labels)
  2. Depth Anything V2 (closeness map)
  3. calculate_mischief() (risk score + warning message)
  4. draw_frame() (annotated output image)

Annotated images are saved to outputs/visualizations/.

Run AFTER scripts/03_train.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
from tqdm import tqdm

from model.detector import fill_depths, infer_depth, infer_yolo, load_depth_model, load_yolo
from model.mischief import calculate_mischief
from model.visualize import draw_frame

ROOT    = Path(__file__).parent.parent
WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
IMG_DIR = ROOT / "data" / "dataset" / "test" / "images"
OUT_DIR = ROOT / "outputs" / "visualizations"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}\nRun scripts/03_train.py first."
        )

    device = get_device()
    print(f"Device : {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO ...")
    yolo = load_yolo(str(WEIGHTS))

    print("Loading Depth Anything V2 ...")
    processor, depth_model = load_depth_model(device)

    image_paths = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.jpeg")) + sorted(IMG_DIR.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMG_DIR}")

    counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for img_path in tqdm(image_paths, desc="Evaluating test images"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path.name}")
            continue

        h, w = frame.shape[:2]

        detections = infer_yolo(yolo, frame)
        depth_map  = infer_depth(processor, depth_model, frame, device)
        detections = fill_depths(detections, depth_map, h, w)
        result     = calculate_mischief(detections, w, h, source=img_path.name)

        counts[result.risk_level] += 1

        annotated = draw_frame(frame, result, depth_map=depth_map)
        cv2.imwrite(str(OUT_DIR / img_path.name), annotated)

    total = sum(counts.values())
    print(f"\n=== Mischief Evaluation Summary ===")
    print(f"  Total : {total}")
    for level, count in counts.items():
        pct = 100.0 * count / total if total > 0 else 0.0
        print(f"  {level:<8}: {count:>6}  ({pct:.1f} %)")
    print(f"\nAnnotated images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
