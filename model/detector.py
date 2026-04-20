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
TARGET_CLASSES = {
    "cat",
    "dog",
    "cup",
    "laptop",
    "potted plant",
    "vase",
    "remote",
    "keyboard",
}

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

    Uses the model's own class-name mapping (yolo.names), so this works for:
    - pretrained COCO weights (e.g. yolo11n.pt)
    - custom fine-tuned weights (e.g. best.pt)

    Returns:
        List of Detection objects filtered to TARGET_CLASSES.
        median_depth is set to 0.0 — call fill_depths() afterwards.
    """
    h, w = frame.shape[:2]
    results = yolo(frame, verbose=False, conf=conf)[0]
    detections: list[Detection] = []

    model_names = yolo.names
    if isinstance(model_names, list):
        model_names = {i: name for i, name in enumerate(model_names)}

    for box in results.boxes:
        cls_id = int(box.cls.item())

        if cls_id not in model_names:
            continue

        class_name = str(model_names[cls_id])

        if class_name not in TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            Detection(
                class_id=cls_id,
                class_name=class_name,
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
    Sample the median closeness value from depth_map for each detection's
    bounding box region and return updated Detection objects.
    """
    updated: list[Detection] = []
    for det in detections:
        x1 = max(0, int(det.bbox.x_min * w))
        y1 = max(0, int(det.bbox.y_min * h))
        x2 = min(w, int(det.bbox.x_max * w))
        y2 = min(h, int(det.bbox.y_max * h))
        roi = depth_map[y1:y2, x1:x2]
        med = float(np.median(roi)) if roi.size > 0 else 0.0
        updated.append(det.model_copy(update={"median_depth": med}))
    return updated
