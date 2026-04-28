"""
Core mischief scoring logic.

calculate_mischief() is the single entry point used by both the batch
evaluation script and the real-time video loop.
"""

from __future__ import annotations

import math

import numpy as np

from schema.Data import BoundingBox, Detection, MischiefResult, PairRisk

# Real-world reference sizes (metres, rough median values).
REAL_SIZE_M: dict[str, float] = {
    "cat":          0.35,
    "dog":          0.50,
    "cup":          0.10,
    "vase":         0.25,
    "laptop":       0.35,
    "keyboard":     0.45,
    "potted plant": 0.30,
    "remote":       0.20,
}

# Blend weight for size-based depth proxy vs. raw Depth Anything signal.
# Kept low because the proxy has wide error bars (unusual angles, size variation).
_ALPHA = 0.25

PET_CLASSES = {"cat", "dog"}

# Risk multiplier table — scales the closeness score by pair context
PAIR_MULTIPLIERS: dict[tuple[str, str], float] = {
    ("cat", "cup"):          1.5,
    ("cat", "vase"):         1.5,
    ("cat", "laptop"):       1.3,
    ("cat", "keyboard"):     1.3,
    ("cat", "potted plant"): 1.2,
    ("cat", "remote"):       1.1,
    ("dog", "laptop"):       1.4,
    ("dog", "keyboard"):     1.3,
    ("dog", "cup"):          1.2,
    ("dog", "vase"):         1.2,
    ("dog", "potted plant"): 1.1,
    ("dog", "remote"):       1.0,
}
DEFAULT_MULTIPLIER = 1.0

HIGH_RISK_MESSAGES: dict[tuple[str, str], str] = {
    ("cat", "cup"):          "Mischief Alert! Your cat is plotting against your drink!",
    ("cat", "vase"):         "Mischief Alert! Your cat is eyeing that vase!",
    ("cat", "laptop"):       "Mischief Alert! Cat occupying your laptop!",
    ("cat", "keyboard"):     "Mischief Alert! Cat about to type a novel!",
    ("cat", "potted plant"): "Mischief Alert! Your cat wants to dig in that plant!",
    ("cat", "remote"):       "Mischief Alert! Cat is going for the remote!",
    ("dog", "laptop"):       "Mischief Alert! Dog is going for the laptop!",
    ("dog", "keyboard"):     "Mischief Alert! Dog chewing on keyboard!",
    ("dog", "cup"):          "Mischief Alert! Dog is after your drink!",
    ("dog", "vase"):         "Mischief Alert! Dog tail incoming — protect that vase!",
    ("dog", "potted plant"): "Mischief Alert! Dog is digging in the plant!",
    ("dog", "remote"):       "Mischief Alert! Dog found the remote!",
}
MEDIUM_RISK_MESSAGE = "Caution! Keep an eye on your pet."
LOW_RISK_MESSAGE    = "All clear. Pet is behaving peacefully."

# 2D component weights (depth is now a multiplicative gate, not an additive term)
W1 = 0.7  # 2D proximity
W3 = 0.3  # contact likelihood

# Edge-gap decay threshold in pixels
GAP_DECAY_PX = 50.0


def _bbox_to_pixels(
    bbox: BoundingBox, w: int, h: int
) -> tuple[int, int, int, int]:
    return (
        int(bbox.x_min * w),
        int(bbox.y_min * h),
        int(bbox.x_max * w),
        int(bbox.y_max * h),
    )


def _proximity_2d(a: Detection, b: Detection) -> float:
    """
    Normalized 2D centroid distance score.

    Returns 1.0 when centres coincide, 0.0 when at opposite corners of
    the image (diagonal distance = sqrt(2) in [0,1] normalized space).
    """
    cx_a = (a.bbox.x_min + a.bbox.x_max) / 2
    cy_a = (a.bbox.y_min + a.bbox.y_max) / 2
    cx_b = (b.bbox.x_min + b.bbox.x_max) / 2
    cy_b = (b.bbox.y_min + b.bbox.y_max) / 2
    dist = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / math.sqrt(2)
    return 1.0 - min(dist, 1.0)


def _size_depth_proxy(det: Detection, img_w: int, img_h: int) -> float | None:
    """
    Metric depth proxy from bounding-box pixel area and known real-world size.

    Returns a farness score in [0, 1] (1 = very far), or None when the
    class has no size entry.
    """
    real_m = REAL_SIZE_M.get(det.class_name)
    if real_m is None:
        return None
    px_w = (det.bbox.x_max - det.bbox.x_min) * img_w
    px_h = (det.bbox.y_max - det.bbox.y_min) * img_h
    px_size = math.sqrt(max(1.0, px_w * px_h))
    return min((real_m / px_size) / 0.05, 1.0)


def _depth_similarity(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
    """
    Blended depth similarity: Depth Anything ordinal signal + size-corrected proxy.

    Returns 1.0 when objects share the same depth plane, 0.0 at maximum
    separation. Acts as a multiplicative gate in calculate_mischief so that
    objects at different depths suppress the 2D-driven score.
    """
    da_sim = 1.0 - abs(a.median_depth - b.median_depth)

    proxy_a = _size_depth_proxy(a, img_w, img_h)
    proxy_b = _size_depth_proxy(b, img_w, img_h)

    if proxy_a is None or proxy_b is None:
        return da_sim

    size_sim = 1.0 - abs(proxy_a - proxy_b)
    return _ALPHA * size_sim + (1.0 - _ALPHA) * da_sim


def _contact_likelihood(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
    """
    Combined IoU + edge-proximity bonus.

    Returns 1.0 when boxes overlap fully, decays as boxes move apart.
    The edge bonus is 1.0 when edges touch and 0.0 at GAP_DECAY_PX pixels.
    """
    ax1, ay1, ax2, ay2 = _bbox_to_pixels(a.bbox, img_w, img_h)
    bx1, by1, bx2, by2 = _bbox_to_pixels(b.bbox, img_w, img_h)

    # IoU
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    iou = inter / (area_a + area_b - inter)

    # Pixel gap between nearest edges
    gap_x = float(max(0, max(bx1 - ax2, ax1 - bx2)))
    gap_y = float(max(0, max(by1 - ay2, ay1 - by2)))
    gap_px = math.sqrt(gap_x ** 2 + gap_y ** 2)
    edge_bonus = max(0.0, 1.0 - gap_px / GAP_DECAY_PX)

    return min(1.0, iou + edge_bonus)


def calculate_mischief(
    detections: list[Detection],
    depth_map: np.ndarray,
    img_w: int,
    img_h: int,
    source: str = "unknown",
) -> MischiefResult:
    """
    Compute a mischief risk score for every (pet, object) pair in the frame.

    Args:
        detections: YOLO detections with filled median_depth values.
        depth_map:  Closeness map (H, W) float32 — used structurally but
                    individual pixel values are already in each Detection.
        img_w:      Frame width in pixels (for edge-gap calculation).
        img_h:      Frame height in pixels.
        source:     Identifier for logging (filename or "video_frame").

    Returns:
        MischiefResult with pairs sorted by risk_score descending.
    """
    pets = [d for d in detections if d.class_name in PET_CLASSES]
    objs = [d for d in detections if d.class_name not in PET_CLASSES]

    pairs: list[PairRisk] = []
    for pet in pets:
        for obj in objs:
            prox    = _proximity_2d(pet, obj)
            depth   = _depth_similarity(pet, obj, img_w, img_h)
            contact = _contact_likelihood(pet, obj, img_w, img_h)
            # Depth is a multiplicative gate: different depths collapse the score
            # regardless of 2D overlap, preventing false positives from screen geometry.
            closeness = (W1 * prox + W3 * contact) * depth
            multiplier = PAIR_MULTIPLIERS.get(
                (pet.class_name, obj.class_name), DEFAULT_MULTIPLIER
            )
            risk_score = closeness * multiplier
            pairs.append(
                PairRisk(
                    pet=pet,
                    obj=obj,
                    proximity_2d=round(prox, 4),
                    depth_similarity=round(depth, 4),
                    contact_likelihood=round(contact, 4),
                    closeness_score=round(closeness, 4),
                    risk_multiplier=multiplier,
                    risk_score=round(risk_score, 4),
                )
            )

    pairs.sort(key=lambda p: p.risk_score, reverse=True)
    max_risk = pairs[0].risk_score if pairs else 0.0

    if max_risk > 0.65:
        risk_level = "HIGH"
        key = (pairs[0].pet.class_name, pairs[0].obj.class_name)
        warning = HIGH_RISK_MESSAGES.get(
            key, f"Mischief Alert! {key[0]} is near the {key[1]}!"
        )
    elif max_risk > 0.3:
        risk_level = "MEDIUM"
        warning = MEDIUM_RISK_MESSAGE
    else:
        risk_level = "LOW"
        warning = LOW_RISK_MESSAGE

    return MischiefResult(
        source=source,
        detections=detections,
        pairs=pairs,
        max_risk_score=round(max_risk, 4),
        risk_level=risk_level,
        warning_message=warning,
    )
