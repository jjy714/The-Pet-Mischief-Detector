# Depth-Aware Mischief Scoring Fix

## The Problem

The original scorer combines three components additively:

```
closeness = W1*proximity_2d + W2*depth_similarity + W3*contact_likelihood
           = 0.5            + 0.3                 + 0.2
```

Two failure modes:

**Failure 1 — same-depth-band false match (original concern):**
Two objects at genuinely different real-world distances share the same
Depth Anything ordinal value (e.g. both in the "mid-room" band), so
`_depth_similarity` returns ~1.0 and inflates the score.

**Failure 2 — 2D overlap at different depths (dominant failure):**
A cat close to the camera (large bounding box) and a vase further away
(small bounding box) overlap in 2D screen space, driving W1 and W3 high.
Even with a perfect depth signal returning 0.0, the math is:

```
closeness = 0.5 * 0.9 + 0.3 * 0.0 + 0.2 * 0.5 = 0.55
risk_score = 0.55 * 1.5 (cat,vase) = 0.825  → HIGH  ← wrong
```

The 2D components (70%) outvote depth (30%). Depth has no veto power.

---

## Root Cause

Depth is an *additive* term bounded to W2 = 0.3. Two objects on opposite
sides of the room still score HIGH if their bounding boxes overlap on screen.

---

## The Fix: Three combined changes

### Part 1 — Multiplicative depth gate

Change depth from an additive voter to a multiplicative gate:

```python
closeness = (W1 * prox + W3 * contact) * depth_sim
```

Now depth_sim = 0.5 → entire score halved. Objects at truly different depths
can never score HIGH regardless of 2D geometry.

Revisiting the failure case:
```
depth_sim = 0.5  (cat depth 0.9, vase depth 0.4)
2d_part   = 0.7 * 0.9 + 0.3 * 0.5 = 0.78
closeness = 0.78 * 0.5 = 0.39
risk_score = 0.39 * 1.5 = 0.585  → MEDIUM  ← correct
```

### Part 2 — Size-range-corrected depth proxy

Depth Anything is ordinal only. The pinhole camera model gives a metric proxy:

```
distance_m ∝ real_size_m / pixel_size
```

Using a *range* `(min_m, max_m)` instead of a fixed value captures
per-class variance. The ratio `min_m / max_m` becomes a confidence score
that automatically scales down the proxy's contribution for high-variance
classes (dog ≈ 0.19) without any per-class special-casing.

```python
# (min_m, max_m) in metres
REAL_SIZE_M = {
    "cat":          (0.25, 0.50),
    "dog":          (0.15, 0.80),   # Chihuahua → Great Dane
    "cup":          (0.08, 0.12),
    "vase":         (0.15, 0.45),
    "laptop":       (0.30, 0.38),   # 13"–16" screen widths
    "keyboard":     (0.35, 0.50),
    "potted plant": (0.15, 0.60),
    "remote":       (0.15, 0.25),
}

def _size_depth_proxy(det, img_w, img_h) -> tuple[float, float] | None:
    size_range = REAL_SIZE_M.get(det.class_name)
    if size_range is None:
        return None
    min_m, max_m = size_range
    confidence = min_m / max_m          # narrow range → high confidence
    mid_m = (min_m + max_m) / 2
    px_size = sqrt(max(1.0, px_w * px_h))
    proxy = min((mid_m / px_size) / 0.05, 1.0)
    return proxy, confidence

def _depth_similarity(a, b, img_w, img_h) -> float:
    da_sim = 1.0 - abs(a.median_depth - b.median_depth)
    result_a = _size_depth_proxy(a, img_w, img_h)
    result_b = _size_depth_proxy(b, img_w, img_h)
    if result_a is None or result_b is None:
        return da_sim
    proxy_a, conf_a = result_a
    proxy_b, conf_b = result_b
    effective_alpha = _ALPHA * min(conf_a, conf_b)   # key line
    size_sim = 1.0 - abs(proxy_a - proxy_b)
    return effective_alpha * size_sim + (1.0 - effective_alpha) * da_sim
```

**Confidence values by class:**

| Class | Range (m) | Confidence |
|-------|-----------|------------|
| laptop | 0.30–0.38 | 0.79 |
| cup | 0.08–0.12 | 0.67 |
| remote | 0.15–0.25 | 0.60 |
| cat | 0.25–0.50 | 0.50 |
| vase | 0.15–0.45 | 0.33 |
| potted plant | 0.15–0.60 | 0.25 |
| dog | 0.15–0.80 | **0.19** |

A dog paired with any object gets `effective_alpha = 0.25 × 0.19 = 0.047` —
essentially pure DA signal. A laptop paired with a cup gets `0.25 × 0.67 = 0.17`.

### Part 3 — Raise HIGH threshold

The multiplicative formula produces a tighter score range. Raise the HIGH
threshold from 0.6 → 0.65 to avoid borderline false positives near the
old boundary.

---

## Summary of Changes

| Change | Why |
|--------|-----|
| `closeness = (W1*prox + W3*contact) * depth` | depth becomes a gate, not a vote |
| Size ranges `(min_m, max_m)` replace fixed values | encodes per-class uncertainty |
| `effective_alpha = _ALPHA * min(conf_a, conf_b)` | high-variance classes auto-downweight proxy |
| `_ALPHA = 0.25` (ceiling) | limits max proxy influence even for well-known sizes |
| `W1=0.7, W3=0.3` (W2 removed) | rebalanced for 2-term inner sum |
| HIGH threshold `0.6 → 0.65` | compensates for tighter multiplicative range |

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Depth has veto power over 2D false positives | True close pairs at same depth correctly preserved |
| Range-based confidence eliminates per-class alpha tuning | Unusual angles (top-down) still break the pixel-area assumption |
| Dog pairs automatically fall back to DA signal | Unknown-class objects (no size entry) always use raw DA |
| No extra model or calibration required | `0.05` normalization constant may need tuning for different camera setups |
