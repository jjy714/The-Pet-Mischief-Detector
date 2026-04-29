"""
Core mischief scoring logic.

calculate_mischief() is the single entry point used by both the batch
evaluation script and the real-time video loop.
"""

from __future__ import annotations

import math

from schema.Data import BoundingBox, Detection, MischiefResult, PairRisk

# Real-world size ranges (min_m, max_m).  Midpoint is used as the proxy estimate;
# the ratio min/max becomes a per-class confidence score that scales _ALPHA down
# for high-variance classes (e.g. dog 0.19 vs. laptop 0.84), automatically
# falling back to the raw Depth Anything signal when size is unreliable.
REAL_SIZE_M: dict[str, tuple[float, float]] = {
    "cat":          (0.25, 0.50),  # domestic cats are fairly consistent
    "dog":          (0.15, 0.80),  # Chihuahua → Great Dane
    "cup":          (0.08, 0.12),
    "vase":         (0.15, 0.45),
    "laptop":       (0.30, 0.38),  # 13"–16" screen widths
    "keyboard":     (0.35, 0.50),  # compact → full-size
    "potted plant": (0.15, 0.60),
    "remote":       (0.15, 0.25),
}

# Maximum blend weight for size-based depth proxy vs. raw Depth Anything signal.
# Actual per-pair weight is _ALPHA * min(conf_a, conf_b), so high-variance
# classes (dog) automatically use a much lower effective alpha.
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

def _bbox_to_pixels(
    bbox: BoundingBox, w: int, h: int
) -> tuple[int, int, int, int]:
    return (
        int(bbox.x_min * w),
        int(bbox.y_min * h),
        int(bbox.x_max * w),
        int(bbox.y_max * h),
    )


def _proximity_2d(a: Detection, b: Detection, img_w: int, img_h: int) -> float:
    """
    Edge-gap proximity score normalized by image diagonal.

    Returns 1.0 when boxes touch or overlap, decays to 0.0 as the gap
    approaches the full image diagonal.  Size-aware: a large object whose
    edge is 2 px from another scores higher than a small object whose
    centroid is equally close.
    """
    ax1 = a.bbox.x_min * img_w;  ay1 = a.bbox.y_min * img_h
    ax2 = a.bbox.x_max * img_w;  ay2 = a.bbox.y_max * img_h
    bx1 = b.bbox.x_min * img_w;  by1 = b.bbox.y_min * img_h
    bx2 = b.bbox.x_max * img_w;  by2 = b.bbox.y_max * img_h

    gap_x = max(0.0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0.0, max(by1 - ay2, ay1 - by2))
    gap_px = math.sqrt(gap_x ** 2 + gap_y ** 2)
    diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    return 1.0 - min(gap_px / diagonal, 1.0)


def _size_depth_proxy(
    det: Detection, img_w: int, img_h: int
) -> tuple[float, float] | None:
    """
    Metric depth proxy from bounding-box pixel area and known real-world size.

    Returns (proxy, confidence) where:
      proxy      — farness score in [0, 1] (1 = very far), based on midpoint size
      confidence — min_m / max_m in (0, 1]; narrow range → high confidence
    Returns None when the class has no size entry.
    """
    size_range = REAL_SIZE_M.get(det.class_name)
    if size_range is None:
        return None
    min_m, max_m = size_range
    mid_m = (min_m + max_m) / 2
    confidence = min_m / max_m
    px_w = (det.bbox.x_max - det.bbox.x_min) * img_w
    px_h = (det.bbox.y_max - det.bbox.y_min) * img_h
    px_size = math.sqrt(max(1.0, px_w * px_h))
    proxy = min((mid_m / px_size) / 0.05, 1.0)
    return proxy, confidence


def _depth_similarity(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
    """
    Blended depth similarity: Depth Anything ordinal signal + size-corrected proxy.

    The proxy's blend weight is scaled by min(conf_a, conf_b) so that
    high-variance classes (dog: conf≈0.19) automatically fall back toward
    the raw DA signal, while well-known sizes (laptop: conf≈0.84) get fuller
    proxy weight.

    Acts as a multiplicative gate in calculate_mischief.
    """
    da_sim = 1.0 - abs(a.median_depth - b.median_depth)

    result_a = _size_depth_proxy(a, img_w, img_h)
    result_b = _size_depth_proxy(b, img_w, img_h)

    if result_a is None or result_b is None:
        return da_sim

    proxy_a, conf_a = result_a
    proxy_b, conf_b = result_b
    effective_alpha = _ALPHA * min(conf_a, conf_b)

    size_sim = 1.0 - abs(proxy_a - proxy_b)
    return effective_alpha * size_sim + (1.0 - effective_alpha) * da_sim


def _contact_likelihood(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
    """
    IoU-based contact score.

    Returns 1.0 when boxes overlap fully, 0.0 when they do not overlap.
    Near-but-not-touching proximity is already captured by _proximity_2d
    (edge-gap), so this function only measures actual overlap.
    """
    ax1, ay1, ax2, ay2 = _bbox_to_pixels(a.bbox, img_w, img_h)
    bx1, by1, bx2, by2 = _bbox_to_pixels(b.bbox, img_w, img_h)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _classify(max_risk: float, top_pair: "PairRisk | None") -> tuple[str, str]:
    """
    Map a max risk score to (risk_level, warning_message).

    Factored out so video mode can call it with a temporally smoothed score
    without duplicating the threshold constants.

    Effective per-pair HIGH closeness thresholds (0.65 / multiplier):
      cat+cup / cat+vase (×1.5) → 0.43
      dog+laptop (×1.4)         → 0.46
      cat/dog+keyboard (×1.3)   → 0.50
      cat/dog+cup/vase (×1.2)   → 0.54
      cat/dog+remote (×1.0–1.1) → 0.59–0.65
    Fragile objects intentionally fire at lower closeness.
    """
    if max_risk > 0.65:
        key = (top_pair.pet.class_name, top_pair.obj.class_name) if top_pair else ("", "")
        return "HIGH", HIGH_RISK_MESSAGES.get(
            key, f"Mischief Alert! {key[0]} is near the {key[1]}!"
        )
    elif max_risk > 0.3:
        return "MEDIUM", MEDIUM_RISK_MESSAGE
    return "LOW", LOW_RISK_MESSAGE


def calculate_mischief(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    source: str = "unknown",
) -> MischiefResult:
    """
    Compute a mischief risk score for every (pet, object) pair in the frame.

    Args:
        detections: YOLO detections with filled median_depth values.
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
            prox    = _proximity_2d(pet, obj, img_w, img_h)
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
    top_pair = pairs[0] if pairs else None
    risk_level, warning = _classify(max_risk, top_pair)

    return MischiefResult(
        source=source,
        detections=detections,
        pairs=pairs,
        max_risk_score=round(max_risk, 4),
        risk_level=risk_level,
        warning_message=warning,
    )
