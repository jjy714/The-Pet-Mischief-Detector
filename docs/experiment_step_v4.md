# Experiment Step V4 — Remaining Issue Fix Plan

## History

| Version | Key change |
|---------|------------|
| v1 (Plan.md) | Original additive formula: `W1·prox + W2·depth + W3·contact` |
| v2 (depth_size_fix.md) | Multiplicative depth gate + size-range proxy with confidence-weighted alpha |
| v3 (experiment_step_v3.md) | Portrait strip depth sampling + resolution-relative gap decay + startup gate |
| **v4 (this doc)** | Fix plan for four remaining open issues: C2, S4, M1, M2 |

---

## Open Issues After v3

| ID | Issue | Severity |
|----|-------|----------|
| C2 | Per-frame depth normalization makes values frame-relative | Critical |
| S4 | More objects in frame → higher false positive rate | Significant |
| M1 | Threshold/multiplier asymmetry undocumented | Moderate |
| M2 | `_proximity_2d` uses centroid distance (size-blind) | Moderate |

---

## C2 — EMA Depth Normalization (Video Mode)

### Problem

`infer_depth` normalizes to [0, 1] using only the current frame's min/max:

```python
closeness = 1.0 - (depth - d_min) / (d_max - d_min)
```

The scale shifts every frame as scene content changes. Two equidistant objects
can get opposite ends of the scale in a flat scene. Frame-to-frame depth values
are not comparable, so temporal reasoning is impossible.

### Fix

Split `infer_depth` into two functions:

```python
def _run_depth_model(processor, depth_model, frame, device) -> np.ndarray:
    """Returns raw (un-normalized) depth output, larger = farther."""
    ...  # existing model inference, omit normalization step

def infer_depth(processor, depth_model, frame, device) -> np.ndarray:
    """Per-frame normalization — used in eval mode (unchanged behavior)."""
    raw = _run_depth_model(processor, depth_model, frame, device)
    d_min, d_max = float(raw.min()), float(raw.max())
    if d_max - d_min < 1e-6:
        return np.zeros(raw.shape, dtype=np.float32)
    return (1.0 - (raw - d_min) / (d_max - d_min)).astype(np.float32)
```

Add EMA state to `_SharedState` in `main.py`:

```python
class _SharedState:
    def __init__(self):
        ...
        self._d_min_ema: float | None = None
        self._d_max_ema: float | None = None

    def set_depth_ema(self, raw: np.ndarray, alpha: float = 0.1) -> None:
        d_min = float(raw.min())
        d_max = float(raw.max())
        with self._lock:
            if self._d_min_ema is None:
                self._d_min_ema, self._d_max_ema = d_min, d_max
            else:
                self._d_min_ema = alpha * d_min + (1 - alpha) * self._d_min_ema
                self._d_max_ema = alpha * d_max + (1 - alpha) * self._d_max_ema
            span = self._d_max_ema - self._d_min_ema
            if span < 1e-6:
                self._depth = np.zeros(raw.shape, dtype=np.float32)
            else:
                normed = (raw - self._d_min_ema) / span
                self._depth = (1.0 - np.clip(normed, 0.0, 1.0)).astype(np.float32)
            self._depth_count += 1
```

Depth worker calls `_run_depth_model` + `state.set_depth_ema` instead of
`infer_depth` + `state.set_depth`.

Eval mode calls `infer_depth` unchanged — no behavior change for batch processing.

**Alpha = 0.1**: at 10 FPS depth updates, the EMA time constant is ~10 frames
(~1 second). This smooths normalization across scene transitions without lagging
on genuine camera moves.

---

## S4 — Temporal Hysteresis (Video Mode)

### Problem

`max_risk = pairs[0].risk_score`. With N objects in frame, there are N pairs —
each is an independent false-positive opportunity. A cluttered room fires HIGH
more often than a sparse room regardless of actual behavior. A sleeping cat next
to a vase fires HIGH continuously.

### Fix

Maintain a rolling window of risk scores in `run_video`:

```python
from collections import deque

risk_history: deque[float] = deque(maxlen=15)
```

After computing `result`, append `result.max_risk_score` and classify by the
**minimum** of the window (conservative: require sustained risk):

```python
risk_history.append(result.max_risk_score)
conservative_max = min(risk_history)

if conservative_max > 0.65:
    level, warning = "HIGH", _pick_warning(result)
elif conservative_max > 0.3:
    level, warning = "MEDIUM", MEDIUM_RISK_MESSAGE
else:
    level, warning = "LOW", LOW_RISK_MESSAGE

result = result.model_copy(update={"risk_level": level, "warning_message": warning})
```

**15-frame window at 15–20 FPS display rate ≈ 0.75–1 second.** A cat that walks
next to a vase and immediately away will not trigger HIGH. A cat that sits down
next to a vase for one second will.

To avoid duplicating threshold constants, factor the classification logic out of
`calculate_mischief` into a shared helper:

```python
# mischief.py
def _classify(max_risk: float, top_pair: PairRisk | None) -> tuple[str, str]:
    if max_risk > 0.65:
        key = (top_pair.pet.class_name, top_pair.obj.class_name) if top_pair else ("", "")
        return "HIGH", HIGH_RISK_MESSAGES.get(key, f"Mischief Alert! {key[0]} near {key[1]}!")
    elif max_risk > 0.3:
        return "MEDIUM", MEDIUM_RISK_MESSAGE
    return "LOW", LOW_RISK_MESSAGE
```

`calculate_mischief` and `run_video` both call `_classify`.

This simultaneously addresses C1 (startup false HIGHs also require 15 frames of
sustained score, though C1 is already gated by the `depth_map is None` check).

---

## M1 — Document Threshold/Multiplier Asymmetry

### Problem

Single HIGH threshold (0.65) combined with per-pair multipliers (1.0–1.5)
creates implicit per-pair thresholds that are not obvious from reading the code.

| Pair | Multiplier | Closeness needed for HIGH |
|------|-----------|--------------------------|
| cat + cup | 1.5 | 0.43 |
| cat + vase | 1.5 | 0.43 |
| dog + laptop | 1.4 | 0.46 |
| dog + keyboard | 1.3 | 0.50 |
| cat + keyboard | 1.3 | 0.50 |
| cat + laptop | 1.3 | 0.50 |
| cat + potted plant | 1.2 | 0.54 |
| dog + cup | 1.2 | 0.54 |
| dog + vase | 1.2 | 0.54 |
| cat + remote | 1.1 | 0.59 |
| dog + potted plant | 1.1 | 0.59 |
| dog + remote | 1.0 | **0.65** |

### Fix

Add a comment block immediately above the `if max_risk > 0.65:` line in
`calculate_mischief`:

```python
# Effective per-pair HIGH closeness thresholds (0.65 / multiplier):
#   cat+cup / cat+vase (×1.5) → 0.43
#   dog+laptop (×1.4)         → 0.46
#   cat/dog+keyboard (×1.3)   → 0.50
#   cat/dog+cup/vase (×1.2)   → 0.54
#   cat/dog+remote (×1.0–1.1) → 0.59–0.65
# Fragile objects intentionally fire at lower closeness.
if max_risk > 0.65:
```

No logic change — documentation only.

---

## M2 — Edge-Gap Proximity (Replace Centroid Distance)

### Problem

`_proximity_2d` computes centroid-to-centroid distance. This ignores object size:
a large cat whose edge is 2 px from a cup can have the same centroid distance as
a small cat 10 cm away. The `contact_likelihood` edge bonus partially compensates,
but the dominant W1 = 0.7 proximity term does not account for size at all.

### Fix

Replace centroid distance with **edge-to-edge gap normalized by image diagonal**.
When boxes overlap or touch, gap = 0 → score = 1.0 (maximum). When boxes are at
opposite corners, gap ≈ diagonal → score ≈ 0.0.

```python
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
```

New signature adds `img_w, img_h`. Update the call in `calculate_mischief`:
```python
prox = _proximity_2d(pet, obj, img_w, img_h)
```

Since edge-gap already captures the "boxes are touching" signal, **simplify
`_contact_likelihood` to IoU only** (remove `edge_bonus` and `_GAP_DECAY_FRAC`):

```python
def _contact_likelihood(a: Detection, b: Detection, img_w: int, img_h: int) -> float:
    ax1, ay1, ax2, ay2 = _bbox_to_pixels(a.bbox, img_w, img_h)
    bx1, by1, bx2, by2 = _bbox_to_pixels(b.bbox, img_w, img_h)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)
```

`_GAP_DECAY_FRAC` constant can be removed entirely.

**Stale docstring**: the `_contact_likelihood` docstring mentions "edge-proximity
bonus" — update it to reflect IoU-only behavior when M2 is applied.

**Risk check**: run `scripts/05_mischief_eval.py` before and after. If HIGH count
increases > 20% (edge-gap makes nearby non-overlapping objects score higher), raise
the HIGH threshold 0.65 → 0.70.

---

## Implementation Order

| Step | Change | File(s) | Risk |
|------|--------|---------|------|
| 1 | M1: Add threshold comment block | `mischief.py` | None — docs only |
| 2 | C2: Split `infer_depth` + EMA state | `detector.py`, `main.py` | Low — eval path unchanged |
| 3 | S4: Temporal hysteresis + `_classify` helper | `mischief.py`, `main.py` | Low — video only |
| 4 | M2: Edge-gap proximity + IoU-only contact | `mischief.py` | Medium — recalibrate threshold if needed |

---

## Expected Impact

| Issue | Before fix | After fix |
|-------|-----------|-----------|
| C2 | Depth scale shifts every frame; flat scene → noise amplified | EMA stabilizes scale; flat scene stays flat |
| S4 | Cluttered room → persistent false HIGHs | 15-frame window requires ~1 s sustained contact |
| M1 | Asymmetric thresholds invisible in code | Per-pair effective thresholds documented inline |
| M2 | Size-blind centroid distance (large dog scores same as small cat) | Edge-gap: large objects only gain if their edges are actually close |
