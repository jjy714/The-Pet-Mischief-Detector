from __future__ import annotations

import numpy as np
import cv2

from schema.Data import Detection, MischiefResult

PET_CLASSES = {"cat", "dog"}

# BGR colour map for each risk level
RISK_COLOURS: dict[str, tuple[int, int, int]] = {
    "HIGH":   (0,   0,   200),
    "MEDIUM": (0,   140, 255),
    "LOW":    (40,  160,  40),
}
PET_BOX_COLOUR: tuple[int, int, int] = (50,  180, 255)
OBJ_BOX_COLOUR: tuple[int, int, int] = (200,  80,  40)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.5
BOX_THICK   = 2
TEXT_THICK  = 1

BANNER_HEIGHT = 44
DEPTH_THUMB_FRAC = 0.25


## return the pixel-space centre of a detection bounding box
def _pixel_centre(det: Detection, w: int, h: int) -> tuple[int, int]:
    cx = int((det.bbox.x_min + det.bbox.x_max) / 2 * w)
    cy = int((det.bbox.y_min + det.bbox.y_max) / 2 * h)
    return cx, cy


## draw a dashed line between pt1 and pt2 in-place
def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 12,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = max(1, int(np.hypot(x2 - x1, y2 - y1)))
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    draw, seg = True, 0
    while seg < length:
        end = min(seg + dash_len, length)
        if draw:
            p1 = (int(x1 + seg * dx), int(y1 + seg * dy))
            p2 = (int(x1 + end * dx), int(y1 + end * dy))
            cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)
        seg = end
        draw = not draw


## render bounding boxes, risk banner, pair connector, and depth inset onto a copy of the input frame
def draw_frame(
    frame: np.ndarray,
    result: MischiefResult,
    depth_map: np.ndarray | None = None,
    fps_display: float | None = None,
    fps_depth: float | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    risk_colour = RISK_COLOURS[result.risk_level]

    for det in result.detections:
        x1 = int(det.bbox.x_min * w)
        y1 = int(det.bbox.y_min * h)
        x2 = int(det.bbox.x_max * w)
        y2 = int(det.bbox.y_max * h)
        colour = PET_BOX_COLOUR if det.class_name in PET_CLASSES else OBJ_BOX_COLOUR
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, BOX_THICK)

        label = f"{det.class_name} {det.confidence:.2f}"
        (lw, lh), baseline = cv2.getTextSize(label, FONT, LABEL_SCALE, TEXT_THICK)
        label_y1 = max(0, y1 - lh - baseline - 4)
        cv2.rectangle(out, (x1, label_y1), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - baseline - 2),
            FONT, LABEL_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA,
        )

    if result.pairs:
        top = result.pairs[0]
        _draw_dashed_line(
            out,
            _pixel_centre(top.pet, w, h),
            _pixel_centre(top.obj, w, h),
            risk_colour,
            thickness=BOX_THICK,
        )

    cv2.rectangle(out, (0, 0), (w, BANNER_HEIGHT), risk_colour, -1)
    banner_text = f"{result.risk_level} RISK  |  {result.warning_message}"
    cv2.putText(
        out, banner_text,
        (8, BANNER_HEIGHT - 12),
        FONT, LABEL_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA,
    )

    if depth_map is not None:
        thumb_w = max(1, int(w * DEPTH_THUMB_FRAC))
        thumb_h = max(1, int(h * DEPTH_THUMB_FRAC))
        depth_coloured = cv2.applyColorMap(
            (depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
        )
        thumb = cv2.resize(depth_coloured, (thumb_w, thumb_h))
        out[h - thumb_h : h, w - thumb_w : w] = thumb
        cv2.putText(
            out, "Depth",
            (w - thumb_w + 4, h - thumb_h + 16),
            FONT, 0.45, (255, 255, 255), TEXT_THICK, cv2.LINE_AA,
        )

    if fps_display is not None:
        cv2.putText(
            out, f"FPS: {fps_display:.1f}",
            (w - 130, BANNER_HEIGHT + 20),
            FONT, 0.48, (200, 200, 200), TEXT_THICK, cv2.LINE_AA,
        )
    if fps_depth is not None:
        cv2.putText(
            out, f"Depth FPS: {fps_depth:.1f}",
            (w - 160, BANNER_HEIGHT + 40),
            FONT, 0.48, (200, 200, 200), TEXT_THICK, cv2.LINE_AA,
        )

    return out
