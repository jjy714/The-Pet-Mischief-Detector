#!/usr/bin/env python3
"""
Task 2 — Split and format the dataset.

Reproducibly splits the staged images into train / val / test sets
(80 % / 10 % / 10 %) using a fixed random seed, then generates
schema/dataset.yaml for Ultralytics YOLO training.

Uses symlinks to avoid duplicating data on disk. Falls back to copying
on file-systems that do not support symlinks.

Run AFTER scripts/01_curate_coco.py.
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

ROOT        = Path(__file__).parent.parent
STAGED_DIR  = ROOT / "data" / "staged"
DATASET_DIR = ROOT / "data" / "dataset"
SCHEMA_DIR  = ROOT / "schema"

SEED        = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 1 - TRAIN_RATIO - VAL_RATIO = 0.10

CLASS_NAMES = ["cat", "dog", "cup", "laptop", "potted plant", "vase", "remote", "keyboard"]


def _link_or_copy(src: Path, dst: Path) -> None:
    """Create a symlink from dst → src, or copy if symlinks are unavailable."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def main() -> None:
    staged_images = STAGED_DIR / "images"
    staged_labels = STAGED_DIR / "labels"

    images = sorted(staged_images.iterdir())
    if not images:
        raise FileNotFoundError(
            f"No images found in {staged_images}\n"
            "Run scripts/01_curate_coco.py first."
        )

    random.seed(SEED)
    random.shuffle(images)

    n       = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    # Create split directories
    for split in splits:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    print("Splitting dataset ...")
    for split_name, split_images in splits.items():
        for img_path in split_images:
            lbl_path = staged_labels / (img_path.stem + ".txt")
            _link_or_copy(img_path, DATASET_DIR / split_name / "images" / img_path.name)
            if lbl_path.exists():
                _link_or_copy(lbl_path, DATASET_DIR / split_name / "labels" / lbl_path.name)
        print(f"  {split_name:<5} : {len(split_images):>6} images")

    print(f"\nTotal : {n}  (train={n_train}, val={n_val}, test={n_test})")

    # Write dataset.yaml with absolute path to the dataset root
    yaml_path = SCHEMA_DIR / "dataset.yaml"
    dataset_cfg = {
        "path":  str((DATASET_DIR).resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"dataset.yaml written to {yaml_path}")


if __name__ == "__main__":
    main()
