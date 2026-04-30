#!/usr/bin/env python3

from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pycocotools.coco import COCO
from tqdm import tqdm

# maps COCO category_id to our 0-indexed YOLO class id
COCO_TO_LABEL: dict[int, int] = {
    17: 0,  # cat
    18: 1,  # dog
    47: 2,  # cup
    73: 3,  # laptop
    64: 4,  # potted plant
    86: 5,  # vase
    75: 6,  # remote
    76: 7,  # keyboard
}
CLASS_NAMES = ["cat", "dog", "cup", "laptop", "potted plant", "vase", "remote", "keyboard"]

ROOT       = Path(__file__).parent.parent
RAW_DIR    = ROOT / "data" / "raw"
STAGED_DIR = ROOT / "data" / "staged"


## process one COCO annotation file — copy matching images and write YOLO label files
def process_split(
    ann_file: Path,
    img_dir: Path,
    out_images: Path,
    out_labels: Path,
) -> dict[int, int]:
    if not ann_file.exists():
        print(f"WARNING: {ann_file} not found — skipping.")
        return {}

    coco = COCO(str(ann_file))

    img_to_anns: dict[int, list] = defaultdict(list)
    for coco_cat_id in COCO_TO_LABEL:
        ann_ids = coco.getAnnIds(catIds=[coco_cat_id])
        for ann in coco.loadAnns(ann_ids):
            if ann.get("iscrowd", 0) == 1:
                continue
            img_to_anns[ann["image_id"]].append(ann)

    class_image_counts: dict[int, int] = defaultdict(int)
    skipped = 0

    for img_id, anns in tqdm(img_to_anns.items(), desc=f"  {ann_file.stem}"):
        img_info = coco.imgs[img_id]
        src_ext  = Path(img_info["file_name"]).suffix
        src_path = img_dir / img_info["file_name"]

        if not src_path.exists():
            skipped += 1
            continue

        img_w, img_h = img_info["width"], img_info["height"]
        label_lines: list[str] = []
        classes_seen: set[int] = set()

        for ann in anns:
            coco_cat = ann["category_id"]
            if coco_cat not in COCO_TO_LABEL:
                continue
            label_id = COCO_TO_LABEL[coco_cat]

            ### COCO bbox format is [x_top_left, y_top_left, width, height] — convert to YOLO cx/cy/w/h normalized
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / img_w
            cy = (y + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            nw = min(max(nw, 0.0), 1.0)
            nh = min(max(nh, 0.0), 1.0)

            label_lines.append(f"{label_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            classes_seen.add(label_id)

        if not label_lines:
            continue

        ### zero-padded image_id avoids filename collisions between train and val splits
        stem = f"{img_id:012d}"
        shutil.copy2(src_path, out_images / f"{stem}{src_ext}")
        (out_labels / f"{stem}.txt").write_text("\n".join(label_lines))

        for cls_id in classes_seen:
            class_image_counts[cls_id] += 1

    if skipped:
        print(f"    Skipped {skipped} images (source file missing)")

    return dict(class_image_counts)


## run curation over both COCO train and val splits and save class distribution CSV
def main() -> None:
    out_images = STAGED_DIR / "images"
    out_labels = STAGED_DIR / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total_counts: dict[int, int] = defaultdict(int)

    splits = [
        (RAW_DIR / "annotations" / "instances_train2017.json", RAW_DIR / "train2017"),
        (RAW_DIR / "annotations" / "instances_val2017.json",   RAW_DIR / "val2017"),
    ]

    for ann_file, img_dir in splits:
        print(f"Processing {ann_file.name} ...")
        counts = process_split(ann_file, img_dir, out_images, out_labels)
        for cls_id, n in counts.items():
            total_counts[cls_id] += n

    total_images = len(list(out_images.iterdir()))
    print(f"\n=== Class Distribution (staged) ===")
    rows = []
    for cls_id, name in enumerate(CLASS_NAMES):
        count = total_counts.get(cls_id, 0)
        print(f"  {name:<15} {count:>6} images")
        rows.append({"class_id": cls_id, "class_name": name, "image_count": count})
    print(f"\nTotal staged images: {total_images}")

    csv_path = STAGED_DIR / "class_distribution.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "image_count"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Class distribution saved to {csv_path}")


if __name__ == "__main__":
    main()
