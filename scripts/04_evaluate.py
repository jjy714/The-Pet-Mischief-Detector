#!/usr/bin/env python3
"""
Task 4A — Quantitative evaluation of the trained detector.

Evaluates model/runs/train/weights/best.pt on the held-out test split
and reports:
  - Overall mAP@0.5 and mAP@[0.5:0.95]
  - Precision, Recall, F1
  - Per-class AP@0.5

Metric justification (for the report):
  mAP@0.5        : standard single-threshold metric; easy to interpret and
                   compare across papers.
  mAP@[0.5:0.95] : COCO-style metric that averages over IoU thresholds
                   0.50–0.95; penalises poor bounding-box localisation and
                   is therefore a more rigorous overall quality measure.
  Per-class AP   : surfaces individual weak classes so data augmentation or
                   threshold adjustments can be targeted.

Outputs (outputs/eval/):
  test_metrics.json   — overall metrics as JSON
  per_class_ap.csv    — per-class AP@0.5 table
  per_class_ap.png    — bar chart

Run AFTER scripts/03_train.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT    = Path(__file__).parent.parent
WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
YAML    = ROOT / "schema" / "dataset.yaml"
OUT_DIR = ROOT / "outputs" / "eval"

CLASS_NAMES = ["cat", "dog", "cup", "laptop", "potted plant", "vase", "remote", "keyboard"]


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found: {WEIGHTS}\nRun scripts/03_train.py first."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model   = YOLO(str(WEIGHTS))
    metrics = model.val(data=str(YAML), split="test", verbose=True)

    map50    = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    prec     = float(metrics.box.mp)
    rec      = float(metrics.box.mr)

    print(f"\n=== Test Set Results ===")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@[0.5:0.95] : {map50_95:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")

    per_class_ap50 = metrics.box.ap50.tolist()

    print(f"\n=== Per-Class AP@0.5 ===")
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        ap = per_class_ap50[i] if i < len(per_class_ap50) else 0.0
        print(f"  {name:<15} {ap:.4f}")
        rows.append({"class_id": i, "class_name": name, "ap50": round(ap, 4)})

    # JSON summary
    summary = {
        "map50":          round(map50, 4),
        "map50_95":       round(map50_95, 4),
        "precision":      round(prec, 4),
        "recall":         round(rec, 4),
        "per_class_ap50": {r["class_name"]: r["ap50"] for r in rows},
    }
    json_path = OUT_DIR / "test_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nMetrics saved to {json_path}")

    # CSV
    csv_path = OUT_DIR / "per_class_ap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "ap50"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-class CSV  : {csv_path}")

    # Bar chart
    names   = [r["class_name"] for r in rows]
    ap_vals = [r["ap50"]       for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ap_vals, color="steelblue", edgecolor="black")
    ax.axhline(map50, color="crimson", linestyle="--",
               label=f"Overall mAP@0.5 = {map50:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP@0.5")
    ax.set_title("Per-Class AP@0.5 on Test Set")
    ax.legend()
    for bar, val in zip(bars, ap_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    chart_path = OUT_DIR / "per_class_ap.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart          : {chart_path}")


if __name__ == "__main__":
    main()
