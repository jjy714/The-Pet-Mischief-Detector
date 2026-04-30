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


## load YOLO model from a weights file path
def load_yolo(weights_path: str) -> YOLO:
    return YOLO(weights_path)


## load Depth Anything V2 from HuggingFace and move to the given device
def load_depth_model(device: str = "cpu") -> tuple:
    processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID)
    model.to(device).eval()
    return processor, model


## run YOLO on a BGR frame and return detections filtered to our 8 target classes
def infer_yolo(yolo: YOLO, frame: np.ndarray, conf: float = 0.25) -> list[Detection]:
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


## run Depth Anything V2 and return the raw unnormalized depth array (larger = farther)
def _run_depth_model(
    processor,
    depth_model,
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = depth_model(**inputs)
    raw = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return raw.astype(np.float32)


## convert raw depth to a closeness map where 1 means near and 0 means far
def _normalize_depth(raw: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    h, w = raw.shape
    if d_max - d_min < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    closeness = 1.0 - (raw - d_min) / (d_max - d_min)
    return np.clip(closeness, 0.0, 1.0).astype(np.float32)


## run the depth model on one frame and return a normalized closeness map
def infer_depth(
    processor,
    depth_model,
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    raw = _run_depth_model(processor, depth_model, frame, device)
    return _normalize_depth(raw, float(raw.min()), float(raw.max()))


## sample depth from each bounding box using a portrait strip and store the result in each Detection
def fill_depths(
    detections: list[Detection],
    depth_map: np.ndarray,
    h: int,
    w: int,
) -> list[Detection]:
    updated: list[Detection] = []
    for det in detections:
        x1 = max(0, int(det.bbox.x_min * w))
        y1 = max(0, int(det.bbox.y_min * h))
        x2 = min(w, int(det.bbox.x_max * w))
        y2 = min(h, int(det.bbox.y_max * h))
        bw = x2 - x1
        bh = y2 - y1

        ### narrow center strip avoids sampling the background pixels that surround the object inside the full bounding box
        if bw >= 20:
            strip_w = max(4, bw // 5)
            cx = (x1 + x2) // 2
            sx1 = max(x1, cx - strip_w // 2)
            sx2 = min(x2, cx + strip_w // 2)
            sy1 = y1 + bh // 8
            roi = depth_map[sy1:y2, sx1:sx2]
            if roi.size > 0:
                ### 25th percentile picks the near side of the object rather than the background-biased median
                med = float(np.percentile(roi, 25))
            else:
                roi_fb = depth_map[y1:y2, x1:x2]
                med = float(np.median(roi_fb)) if roi_fb.size > 0 else 0.0
        else:
            roi = depth_map[y1:y2, x1:x2]
            med = float(np.median(roi)) if roi.size > 0 else 0.0

        updated.append(det.model_copy(update={"median_depth": med}))
    return updated
