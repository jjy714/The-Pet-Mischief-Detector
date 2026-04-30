#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

ROOT            = Path(__file__).parent.parent
DEFAULT_WEIGHTS = ROOT / "model" / "runs" / "train" / "weights" / "best.pt"
DEFAULT_OUT_DIR = ROOT / "outputs" / "eval"
YAML_PATH       = ROOT / "schema" / "dataset.yaml"

CLASS_NAMES = ["cat", "dog", "cup", "laptop", "potted plant", "vase", "remote", "keyboard"]

# remapped IDs (0-7) to COCO IDs
OUR_TO_COCO: dict[int, int] = {0: 15, 1: 16, 2: 41, 3: 63, 4: 58, 5: 75, 6: 65, 7: 66}
COCO_TO_OUR: dict[int, int] = {v: k for k, v in OUR_TO_COCO.items()}
COCO_TARGET_IDS = list(OUR_TO_COCO.values())


## compute IoU of two boxes in xywh-normalized format
def _iou(box1: list[float], box2: list[float]) -> float:
    b1x1, b1y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1x2, b1y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2x1, b2y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2x2, b2y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    ix1, iy1 = max(b1x1, b2x1), max(b1y1, b2y1)
    ix2, iy2 = min(b1x2, b2x2), min(b1y2, b2y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


## compute per-class AP at a single IoU threshold using 101-point COCO interpolation
def _ap_at_iou(
    preds_by_img: dict[str, list[tuple[list[float], float]]],
    gt_by_img:    dict[str, list[list[float]]],
    n_gt:         int,
    iou_thresh:   float,
) -> float:
    if n_gt == 0:
        return 0.0

    all_preds = [
        (img_id, box, conf)
        for img_id, preds in preds_by_img.items()
        for box, conf in preds
    ]
    all_preds.sort(key=lambda x: -x[2])

    matched: dict[str, set[int]] = {img_id: set() for img_id in gt_by_img}
    tp_arr, fp_arr = [], []

    for img_id, pred_box, _ in all_preds:
        gts = gt_by_img.get(img_id, [])
        best_iou, best_j = iou_thresh - 1e-9, -1
        for j, gt_box in enumerate(gts):
            if j in matched.get(img_id, set()):
                continue
            iou = _iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            matched[img_id].add(best_j)
            tp_arr.append(1); fp_arr.append(0)
        else:
            tp_arr.append(0); fp_arr.append(1)

    if not tp_arr:
        return 0.0

    tp_cum = np.cumsum(tp_arr).astype(float)
    fp_cum = np.cumsum(fp_arr).astype(float)
    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum)

    ap = 0.0
    for thr in np.linspace(0, 1, 101):
        prec_at = precisions[recalls >= thr]
        ap += float(prec_at.max()) if len(prec_at) else 0.0
    return ap / 101


## compute both AP@0.5 and AP@[0.5:0.95] for one class
def _ap50_and_ap50_95(
    preds_by_img: dict[str, list[tuple[list[float], float]]],
    gt_by_img:    dict[str, list[list[float]]],
    n_gt:         int,
) -> tuple[float, float]:
    ap50    = _ap_at_iou(preds_by_img, gt_by_img, n_gt, 0.50)
    ap50_95 = float(np.mean([
        _ap_at_iou(preds_by_img, gt_by_img, n_gt, t)
        for t in np.linspace(0.50, 0.95, 10)
    ]))
    return ap50, ap50_95


## run manual inference on pretrained COCO model and compute AP by remapping class IDs to our 0-7 scheme
def eval_coco_pretrained(
    model: YOLO,
    test_img_dir:   Path,
    test_label_dir: Path,
    imgsz:          int = 640,
) -> tuple[list[float], list[float]]:
    preds_by_class: dict[int, dict[str, list]] = {c: {} for c in range(len(CLASS_NAMES))}
    gt_by_class:    dict[int, dict[str, list]] = {c: {} for c in range(len(CLASS_NAMES))}
    n_gt: dict[int, int] = {c: 0 for c in range(len(CLASS_NAMES))}

    img_files = sorted(
        list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    )
    total = len(img_files)
    print(f"[INFO] Running inference on {total} test images...")

    for i, img_path in enumerate(img_files, 1):
        if i % 200 == 0 or i == total:
            print(f"  {i}/{total}")

        img_id = img_path.stem
        label_path = test_label_dir / (img_id + ".txt")

        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                c = int(parts[0])
                box = [float(x) for x in parts[1:5]]
                gt_by_class[c].setdefault(img_id, []).append(box)
                n_gt[c] += 1

        results = model.predict(str(img_path), classes=COCO_TARGET_IDS, imgsz=imgsz, verbose=False)

        for result in results:
            if result.boxes is None:
                continue
            h, w = result.orig_shape
            for box_data in result.boxes:
                coco_id = int(box_data.cls.item())
                our_id  = COCO_TO_OUR.get(coco_id)
                if our_id is None:
                    continue
                conf = float(box_data.conf.item())
                x1, y1, x2, y2 = box_data.xyxy[0].tolist()
                cx  = (x1 + x2) / (2 * w)
                cy  = (y1 + y2) / (2 * h)
                bw  = (x2 - x1) / w
                bh  = (y2 - y1) / h
                preds_by_class[our_id].setdefault(img_id, []).append(([cx, cy, bw, bh], conf))

    per_class_ap50:     list[float] = []
    per_class_ap50_95:  list[float] = []
    for c in range(len(CLASS_NAMES)):
        ap50, ap50_95 = _ap50_and_ap50_95(
            preds_by_class[c], gt_by_class[c], n_gt[c]
        )
        per_class_ap50.append(ap50)
        per_class_ap50_95.append(ap50_95)

    return per_class_ap50, per_class_ap50_95


## evaluate model on test split, save metrics JSON, per-class CSV, and bar chart
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detector on test split.")
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help="Path to model weights (default: model/runs/train/weights/best.pt)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640). For Exp A use 1280; for A' use 640 with 1280-trained weights.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Directory for output files (default: outputs/eval/)",
    )
    args = parser.parse_args()

    weights = args.weights
    imgsz   = args.imgsz
    out_dir = args.out_dir

    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run scripts/03_train.py first, or pass --weights <path>."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    model    = YOLO(str(weights))
    model_nc = len(model.names)
    is_coco  = model_nc == 80 and model.names.get(0) == "person"

    with open(YAML_PATH) as f:
        ds = yaml.safe_load(f)

    if is_coco:
        print(f"[INFO] {weights.name} is a pretrained COCO model ({model_nc} classes), imgsz={imgsz}.")
        base          = Path(ds["path"])
        test_img_dir  = base / ds["test"]
        test_lbl_dir  = test_img_dir.parent / "labels"
        per_ap50, per_ap50_95 = eval_coco_pretrained(model, test_img_dir, test_lbl_dir, imgsz=imgsz)
        map50    = float(np.mean(per_ap50))
        map50_95 = float(np.mean(per_ap50_95))
        prec = rec = float("nan")
    else:
        if model_nc != len(CLASS_NAMES):
            print(
                f"[WARNING] Model has {model_nc} classes but dataset expects "
                f"{len(CLASS_NAMES)}. Results may be unreliable."
            )
        metrics  = model.val(data=str(YAML_PATH), split="test", imgsz=imgsz, verbose=True)
        per_ap50 = metrics.box.ap50.tolist()
        per_ap50_95 = [float(ap) for ap in metrics.box.ap]
        map50    = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        prec     = float(metrics.box.mp)
        rec      = float(metrics.box.mr)

    print(f"\n=== Test Set Results ({weights.name}, imgsz={imgsz}) ===")
    print(f"  mAP@0.5        : {map50:.4f}")
    print(f"  mAP@[0.5:0.95] : {map50_95:.4f}")
    if not (isinstance(prec, float) and prec != prec):
        print(f"  Precision       : {prec:.4f}")
        print(f"  Recall          : {rec:.4f}")

    print(f"\n=== Per-Class AP@0.5 ===")
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        ap = per_ap50[i] if i < len(per_ap50) else 0.0
        print(f"  {name:<15} {ap:.4f}")
        rows.append({"class_id": i, "class_name": name, "ap50": round(ap, 4)})

    summary: dict = {
        "weights":        str(weights),
        "imgsz":          imgsz,
        "map50":          round(map50, 4),
        "map50_95":       round(map50_95, 4),
        "per_class_ap50": {r["class_name"]: r["ap50"] for r in rows},
    }
    if not (isinstance(prec, float) and prec != prec):
        summary["precision"] = round(prec, 4)
        summary["recall"]    = round(rec, 4)

    json_path = out_dir / "test_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nMetrics saved to {json_path}")

    csv_path = out_dir / "per_class_ap.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "ap50"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-class CSV  : {csv_path}")

    names   = [r["class_name"] for r in rows]
    ap_vals = [r["ap50"]       for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, ap_vals, color="steelblue", edgecolor="black")
    ax.axhline(map50, color="crimson", linestyle="--",
               label=f"Overall mAP@0.5 = {map50:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AP@0.5")
    ax.set_title(f"Per-Class AP@0.5 — {weights.name} @ imgsz={imgsz}")
    ax.legend()
    for bar, val in zip(bars, ap_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=8,
        )
    plt.tight_layout()
    chart_path = out_dir / "per_class_ap.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Chart          : {chart_path}")


if __name__ == "__main__":
    main()
