from __future__ import annotations

import math

from schema.Data import BoundingBox, Detection, MischiefResult, PairRisk

# real-world size ranges used to estimate metric depth from bounding box pixel size
REAL_SIZE_M: dict[str, tuple[float, float]] = {
    "cat":          (0.25, 0.50),
    "dog":          (0.15, 0.80),
    "cup":          (0.08, 0.12),
    "vase":         (0.15, 0.45),
    "laptop":       (0.30, 0.38),
    "keyboard":     (0.35, 0.50),
    "potted plant": (0.15, 0.60),
    "remote":       (0.15, 0.25),
}

### actual blend weight per pair is _ALPHA * min(conf_a, conf_b), so high-variance classes like dog automatically get a much lower effective alpha and fall back to the raw Depth Anything signal
_ALPHA = 0.25

PET_CLASSES = {"cat", "dog"}

# risk multiplier table — higher multiplier means the pair fires HIGH at a lower closeness threshold
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

### depth is a multiplicative gate on the 2D score, not an additive term — different depth planes collapse the score to near zero regardless of 2D proximity
W1 = 0.7
W3 = 0.3

## convert bounding box from normalized coords to pixel coords
def _bbox_to_pixels(
    bbox: BoundingBox, w: int, h: int
) -> tuple[int, int, int, int]:
    return (
        int(bbox.x_min * w),
        int(bbox.y_min * h),
        int(bbox.x_max * w),
        int(bbox.y_max * h),
    )


## compute edge-to-edge gap between two boxes normalized by the image diagonal (1.0 = touching, 0.0 = opposite corners)
def _proximity_2d(a: Detection, b: Detection, img_w: int, img_h: int) -> float:
    ax1 = a.bbox.x_min * img_w;  ay1 = a.bbox.y_min * img_h
    ax2 = a.bbox.x_max * img_w;  ay2 = a.bbox.y_max * img_h
    bx1 = b.bbox.x_min * img_w;  by1 = b.bbox.y_min * img_h
    bx2 = b.bbox.x_max * img_w;  by2 = b.bbox.y_max * img_h

    gap_x = max(0.0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0.0, max(by1 - ay2, ay1 - by2))
    gap_px = math.sqrt(gap_x ** 2 + gap_y ** 2)
    diagonal = math.sqrt(img_w ** 2 + img_h ** 2)
    return 1.0 - min(gap_px / diagonal, 1.0)


## estimate metric depth from bounding box pixel area and the known real-world size range of the class
def _size_depth_proxy(
    det: Detection, img_w: int, img_h: int
) -> tuple[float, float] | None:
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


## blend Depth Anything ordinal depth with a size-based metric proxy to get a depth similarity score
def _depth_similarity(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
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


## compute IoU between two bounding boxes as a contact score (1.0 = full overlap, 0.0 = no overlap)
def _contact_likelihood(
    a: Detection, b: Detection, img_w: int, img_h: int
) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_pixels(a.bbox, img_w, img_h)
    bx1, by1, bx2, by2 = _bbox_to_pixels(b.bbox, img_w, img_h)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


## map a risk score to a risk level string and warning message
def _classify(max_risk: float, top_pair: "PairRisk | None") -> tuple[str, str]:
    ### effective HIGH thresholds differ per pair: cat+cup fires at 0.65/1.5=0.43, dog+remote at 0.65/1.0=0.65 — fragile objects intentionally fire at lower closeness
    if max_risk > 0.65:
        key = (top_pair.pet.class_name, top_pair.obj.class_name) if top_pair else ("", "")
        return "HIGH", HIGH_RISK_MESSAGES.get(
            key, f"Mischief Alert! {key[0]} is near the {key[1]}!"
        )
    elif max_risk > 0.3:
        return "MEDIUM", MEDIUM_RISK_MESSAGE
    return "LOW", LOW_RISK_MESSAGE


## score every pet-object pair in the frame and return the overall risk result
def calculate_mischief(
    detections: list[Detection],
    img_w: int,
    img_h: int,
    source: str = "unknown",
) -> MischiefResult:
    pets = [d for d in detections if d.class_name in PET_CLASSES]
    objs = [d for d in detections if d.class_name not in PET_CLASSES]

    pairs: list[PairRisk] = []
    for pet in pets:
        for obj in objs:
            prox    = _proximity_2d(pet, obj, img_w, img_h)
            depth   = _depth_similarity(pet, obj, img_w, img_h)
            contact = _contact_likelihood(pet, obj, img_w, img_h)
            ### depth acts as a multiplicative gate: objects on different depth planes collapse the score to near zero, preventing false positives from 2D screen geometry
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
