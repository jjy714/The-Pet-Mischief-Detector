"""
Pet Mischief Detector — unified entry point (eval mode only).

Batch-processes a folder of static images, saves annotated outputs to disk.
Used for Task 4B report visualizations and the bonus generalization test.

Usage:
  uv run main.py [--input PATH] [--output PATH]

Options:
  --weights          Path to YOLO weights  (default: model/runs/train/weights/best.pt)
  --input            Image folder to evaluate
                     (default: data/dataset/test/images)
  --output           Output folder for annotated images
                     (default: outputs/visualizations)
  --detector         heuristic | hybrid  (default: heuristic)
  --hybrid-checkpoint  Path to hybrid model checkpoint
                     (default: models/hybrid/checkpoints/best.pt)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).parent

from models.detector import (
    fill_depths,
    infer_depth,
    infer_yolo,
    load_depth_model,
    load_yolo,
)
from models.mischief import _classify, calculate_mischief
from models.hybrid.inference import load_hybrid_model, predict_image as hybrid_predict_image
from schema.Data import MischiefResult
from models.visualize import draw_frame

DEFAULT_WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
DEFAULT_INPUT   = ROOT / "data" / "dataset" / "test" / "images"
DEFAULT_OUTPUT  = ROOT / "outputs" / "visualizations"

_LEVEL_SCORE = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.1}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace, yolo, processor, depth_model, device: str, hybrid_models=None) -> None:
    input_dir = Path(args.input)
    out_dir   = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        sorted(input_dir.glob("*.jpg"))
        + sorted(input_dir.glob("*.jpeg"))
        + sorted(input_dir.glob("*.png"))
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(image_paths)} images from {input_dir} ...")
    counts: dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARNING: could not read {img_path.name}")
            continue

        h, w = frame.shape[:2]
        detections = infer_yolo(yolo, frame)
        depth_map  = infer_depth(processor, depth_model, frame, device)
        detections = fill_depths(detections, depth_map, h, w)
        result     = calculate_mischief(detections, w, h, source=img_path.name)

        if hybrid_models is not None:
            backbone, hyb_model = hybrid_models
            hyb_level = hybrid_predict_image(backbone, hyb_model, frame, detections, device)
            top_pair  = result.pairs[0] if result.pairs else None
            _, warning = _classify(_LEVEL_SCORE[hyb_level], top_pair)
            result = result.model_copy(update={"risk_level": hyb_level, "warning_message": warning})

        counts[result.risk_level] += 1
        annotated = draw_frame(frame, result, depth_map=depth_map)
        cv2.imwrite(str(out_dir / img_path.name), annotated)
        print(f"  [{result.risk_level:>6}]  {img_path.name}  —  {result.warning_message}")

    total = sum(counts.values())
    print(f"\n=== Summary ===")
    for level, count in counts.items():
        pct = 100.0 * count / total if total else 0.0
        print(f"  {level:<8}: {count:>5}  ({pct:.1f} %)")
    print(f"Output saved to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pet Mischief Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run main.py\n"
            "  uv run main.py --input my_photos/ --output outputs/bonus\n"
            "  uv run main.py --detector hybrid\n"
        ),
    )
    parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS),
        help="Path to YOLO weights (default: model/runs/train/weights/best.pt)",
    )
    parser.add_argument(
        "--input", default=str(DEFAULT_INPUT),
        help="Folder of input images (default: data/dataset/test/images)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help="Output folder for annotated images (default: outputs/visualizations)",
    )
    parser.add_argument(
        "--detector", default="heuristic", choices=["heuristic", "hybrid"],
        help="heuristic: rule-based scoring (default); hybrid: CNN+GNN static scoring",
    )
    parser.add_argument(
        "--hybrid-checkpoint",
        default=str(ROOT / "models" / "hybrid" / "checkpoints" / "best.pt"),
        help="[hybrid] Path to hybrid model checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args   = _parse_args()
    device = get_device()
    print(f"Device  : {device}")

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

    hybrid_models = None
    if args.detector == "hybrid":
        ckpt = Path(args.hybrid_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Hybrid checkpoint not found: {ckpt}\n"
                "Run scripts/07_train_hybrid.py --phase precompute then --phase train first."
            )
        print("Loading Hybrid CNN+GNN model ...")
        hybrid_models = load_hybrid_model(str(ckpt), device)

    run_eval(args, yolo, processor, depth_model, device, hybrid_models=hybrid_models)


if __name__ == "__main__":
    main()
