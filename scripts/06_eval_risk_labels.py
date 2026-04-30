#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch

from model.detector import fill_depths, infer_depth, infer_yolo, load_depth_model, load_yolo
from model.mischief import calculate_mischief


ROOT = Path(__file__).parent.parent

LABEL_MAP = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

LEVELS = ["LOW", "MEDIUM", "HIGH"]


## pick GPU if available, else CPU
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


## read a label file and return the highest risk level found on any line
def read_ground_truth(label_path: Path) -> str:
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty label file: {label_path}")

    ### label format is one integer per line (0=LOW, 1=MEDIUM, 2=HIGH); multiple lines take the max
    risk_ids = [int(line.split()[0]) for line in lines]
    max_risk = max(risk_ids)

    if max_risk not in LABEL_MAP:
        raise ValueError(f"Invalid risk label {max_risk} in {label_path}")

    return LABEL_MAP[max_risk]


## parse CLI args for weights, image directory, label directory, and output CSV path
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--images", default=str(ROOT / "data" / "test" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "data" / "test" / "labels"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        weights_name = Path(args.weights).stem
        args.output = str(ROOT / "outputs" / f"risk_eval_{weights_name}.csv")

    return args


## run the full pipeline on each image, compare predictions to ground truth, and save a confusion matrix and CSV
def main() -> None:
    args = parse_args()

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading YOLO: {args.weights}")
    yolo = load_yolo(args.weights)

    print("Loading Depth Anything V2 ...")
    processor, depth_model = load_depth_model(device)

    image_paths = (
        sorted(image_dir.glob("*.jpg"))
        + sorted(image_dir.glob("*.jpeg"))
        + sorted(image_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    rows = []
    confusion = {gt: {pred: 0 for pred in LEVELS} for gt in LEVELS}
    correct = 0
    total = 0

    for img_path in image_paths:
        label_path = label_dir / f"{img_path.stem}.txt"
        gt = read_ground_truth(label_path)

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"WARNING: could not read {img_path.name}")
            continue

        h, w = frame.shape[:2]

        detections = infer_yolo(yolo, frame)
        depth_map = infer_depth(processor, depth_model, frame, device)
        detections = fill_depths(detections, depth_map, h, w)
        result = calculate_mischief(detections, w, h, source=img_path.name)

        pred = result.risk_level
        is_correct = gt == pred

        correct += int(is_correct)
        total += 1
        confusion[gt][pred] += 1

        rows.append({
            "filename": img_path.name,
            "ground_truth": gt,
            "prediction": pred,
            "correct": int(is_correct),
            "max_risk_score": round(result.max_risk_score, 4),
            "warning_message": result.warning_message,
        })

        print(f"[{pred:>6}] GT={gt:<6} {img_path.name} | {result.warning_message}")

    accuracy = correct / total if total else 0.0

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ground_truth",
                "prediction",
                "correct",
                "max_risk_score",
                "warning_message",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Risk Evaluation Summary ===")
    print(f"Total    : {total}")
    print(f"Correct  : {correct}")
    print(f"Accuracy : {accuracy:.4f}")

    print("\n=== Confusion Matrix ===")
    print("GT \\ Pred   LOW   MEDIUM   HIGH")
    for gt in LEVELS:
        print(
            f"{gt:<9} "
            f"{confusion[gt]['LOW']:>4} "
            f"{confusion[gt]['MEDIUM']:>8} "
            f"{confusion[gt]['HIGH']:>6}"
        )

    print(f"\nCSV saved to {output_path}")


if __name__ == "__main__":
    main()
