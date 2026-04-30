#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from models.detector import (
    fill_depths,
    infer_depth,
    infer_yolo,
    load_depth_model,
    load_yolo,
)
from models.gnn.dataset import LABEL_MAP
from models.hybrid.backbone import ResNetBackbone
from models.hybrid.graph_builder import build_static_graph
from models.hybrid.inference import preprocess_frame
from models.hybrid.roi_pool import extract_roi_features
from models.hybrid.train import train_hybrid

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
CLIPS_JSON = ROOT / "data" / "clips" / "clips.json"
IMAGES_DIR = ROOT / "data" / "clips" / "images"
CACHE_DIR = ROOT / "data" / "hybrid_cache"
CKPT_DIR = ROOT / "models" / "hybrid" / "checkpoints"


## pick GPU if available, else CPU
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


## resolve a frame_id to a Path by trying .jpg, .jpeg, .png extensions
def _resolve_frame(frame_id: str, images_dir: Path) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    return None


## run YOLO + depth + backbone on each clip's first frame and save a .pt cache file
def precompute(args: argparse.Namespace) -> None:
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
            f"YOLO weights not found: {weights}\nRun scripts/03_train.py first."
        )

    device = get_device()
    print(f"Device: {device}")

    print("Loading YOLO ...")
    yolo = load_yolo(str(weights))
    print("Loading Depth Anything V2 ...")
    processor, depth_model = load_depth_model(device)
    print("Loading ResNet18 backbone ...")
    backbone = ResNetBackbone().to(device)
    backbone.eval()

    images_dir = Path(args.images)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    skipped = 0

    for clip in tqdm(clips, desc="Pre-computing"):
        ### only the first frame of each clip is used — the hybrid model is static, not temporal
        frame_id = clip["frames"][0]
        img_path = _resolve_frame(frame_id, images_dir)

        if img_path is None:
            print(f"  WARNING: frame not found: {frame_id}")
            skipped += 1
            continue

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path}")
            skipped += 1
            continue

        h, w = frame.shape[:2]
        detections = infer_yolo(yolo, frame)
        depth_map = infer_depth(processor, depth_model, frame, device)
        detections = fill_depths(detections, depth_map, h, w)

        img_t = preprocess_frame(frame, device)
        feat_map, global_feat = backbone.extract(img_t)

        roi_feats = extract_roi_features(feat_map, detections)

        graph = build_static_graph(detections, roi_feats=roi_feats)

        label = LABEL_MAP[clip["risk_level"]]
        cache_file = cache_dir / f"{clip['clip_id']}.pt"

        torch.save(
            {
                "global_feat": global_feat.squeeze(0).cpu(),
                "graph": graph.cpu(),
                "label": label,
            },
            cache_file,
        )

    total = len(clips) - skipped
    print(f"Cached {total} samples to {cache_dir}  ({skipped} skipped)")


## load cached features and train HybridMischiefModel
def train(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache)
    if not any(cache_dir.glob("*.pt")):
        raise FileNotFoundError(
            f"No cache files found in {cache_dir}\n"
            "Run this script with --phase precompute first."
        )

    device = get_device()
    print(f"Device: {device}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(CKPT_DIR / "best.pt")

    print(f"Training for {args.epochs} epochs ...")
    train_hybrid(
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_fraction=args.val,
        device=device,
    )


## parse --phase and dispatch to precompute or train
def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CNN+GNN: pre-compute or train")
    parser.add_argument(
        "--phase",
        required=True,
        choices=["precompute", "train"],
        help="precompute: extract and cache features; train: fit HybridMischiefModel",
    )
    parser.add_argument("--clips", default=str(CLIPS_JSON))
    parser.add_argument("--images", default=str(IMAGES_DIR))
    parser.add_argument("--weights", default=str(WEIGHTS), help="YOLO weights")
    parser.add_argument(
        "--cache", default=str(CACHE_DIR), help="Feature cache directory"
    )
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--val", default=0.2, type=float, help="Validation fraction")
    args = parser.parse_args()

    if args.phase == "precompute":
        precompute(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
