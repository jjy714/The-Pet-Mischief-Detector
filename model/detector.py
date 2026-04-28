"""
Wraps YOLO detection and Depth Anything V2 inference into reusable helpers.

Both models are loaded once at startup and reused across frames/images.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from schema.Data import BoundingBox, Detection

CLASS_NAMES = [
    "cat",
    "dog",
    "cup",
    "laptop",
    "potted plant",
    "vase",
    "remote",
    "keyboard",
]

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def load_yolo(weights_path: str) -> YOLO:
    """Load a YOLO model from a weights file path."""
    return YOLO(weights_path)


def load_depth_model(device: str = "cpu") -> tuple:
    """
    Load Depth Anything V2 Small from HuggingFace.

    Returns:
        (processor, model) — pass both to infer_depth().
    """
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    model.to(device).eval()
    return processor, model


def infer_yolo(yolo: YOLO, frame: np.ndarray, conf: float = 0.25) -> list[Detection]:
    """
    Run YOLO on a BGR frame.

    Returns:
        List of Detection objects. median_depth is set to 0.0 — call
        fill_depths() afterwards to populate it from the depth map.
    """
    h, w = frame.shape[:2]
    results = yolo(frame, verbose=False, conf=conf)[0]
    detections: list[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        if cls_id >= len(CLASS_NAMES):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                class_id=cls_id,
                class_name=CLASS_NAMES[cls_id],
                confidence=float(box.conf.item()),
                bbox=BoundingBox(
                    x_min=min(max(x1 / w, 0.0), 1.0),
                    y_min=min(max(y1 / h, 0.0), 1.0),
                    x_max=min(max(x2 / w, 0.0), 1.0),
                    y_max=min(max(y2 / h, 0.0), 1.0),
                ),
                median_depth=0.0,
            )
        )
    return detections


def infer_depth(
    processor,
    depth_model,
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Run Depth Anything V2 on a BGR frame.

    The raw model output is depth (larger = farther). We normalize to [0, 1]
    and invert so the returned closeness map has 1 = near, 0 = far.

    Returns:
        np.ndarray of shape (H, W), dtype float32, values in [0, 1].
    """
    h, w = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    # Upsample predicted depth to original frame resolution
    depth = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()  # (H, W), larger = farther
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    # Normalize and invert → closeness map (1 = near, 0 = far)
    closeness = 1.0 - (depth - d_min) / (d_max - d_min)
    return closeness.astype(np.float32)


def fill_depths(
    detections: list[Detection],
    depth_map: np.ndarray,
    h: int,
    w: int,
) -> list[Detection]:
    """
    Sample depth from a portrait strip (narrow vertical center column) of each
    bounding box using the 25th-percentile (near-side bias).

    Portrait strip: center ~20% of box width, dropping the top 12.5% of height
    to reduce ceiling/wall contamination above the object.  Near-side percentile
    biases the reading toward the object's front surface, maximising the depth
    difference between close and far objects for the multiplicative gate.

    Falls back to full-bbox median when the box is narrower than 20 px (small
    objects at long range where a portrait strip would be too few pixels).
    """
    updated: list[Detection] = []
    for det in detections:
        x1 = max(0, int(det.bbox.x_min * w))
        y1 = max(0, int(det.bbox.y_min * h))
        x2 = min(w, int(det.bbox.x_max * w))
        y2 = min(h, int(det.bbox.y_max * h))
        bw = x2 - x1
        bh = y2 - y1

        if bw >= 20:
            strip_w = max(4, bw // 5)
            cx = (x1 + x2) // 2
            sx1 = max(x1, cx - strip_w // 2)
            sx2 = min(x2, cx + strip_w // 2)
            sy1 = y1 + bh // 8   # drop top 12.5% (ceiling / hat artifacts)
            roi = depth_map[sy1:y2, sx1:sx2]
            if roi.size > 0:
                med = float(np.percentile(roi, 25))
            else:
                roi_fb = depth_map[y1:y2, x1:x2]
                med = float(np.median(roi_fb)) if roi_fb.size > 0 else 0.0
        else:
            roi = depth_map[y1:y2, x1:x2]
            med = float(np.median(roi)) if roi.size > 0 else 0.0

        updated.append(det.model_copy(update={"median_depth": med}))
    return updated
