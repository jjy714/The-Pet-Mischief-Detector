# Depth-Aware Mischief Scoring Fix

## The Problem

The original scorer combines three components additively:

```
closeness = W1*proximity_2d + W2*depth_similarity + W3*contact_likelihood
           = 0.5            + 0.3                 + 0.2
```

Two failure modes:

**Failure 1 — same-depth-band false match (original doc's concern):**
Two objects at genuinely different real-world distances share the same
Depth Anything ordinal value (e.g. both in the "mid-room" band), so
`_depth_similarity` returns ~1.0 and inflates the score.

**Failure 2 — 2D overlap at different depths (actual dominant failure):**
A cat close to the camera (large bounding box) and a vase further away
(small bounding box) overlap in 2D screen space, driving W1 and W3 high.
Even with a perfect depth signal returning 0.0, the math is:

```
closeness = 0.5 * 0.9 + 0.3 * 0.0 + 0.2 * 0.5 = 0.55
risk_score = 0.55 * 1.5 (cat,vase) = 0.825  → HIGH  ← wrong
```

The 2D components (70%) outvote depth (30%). The depth signal has no
veto power — it cannot prevent a false positive caused by 2D overlap.

---

## Root Cause

Depth is an *additive* term. Its contribution is bounded to W2 = 0.3
regardless of how different the depths are. Two objects on opposite sides
of the room can still score HIGH if their bounding boxes overlap on screen.

---

## The Fix: Multiplicative Depth Gate + Size-Corrected Proxy

### Part 1 — Multiplicative depth

Change depth from additive voter to a multiplicative gate:

```python
closeness = (W1 * prox + W3 * contact) * depth_sim
```

Now depth_sim = 0.5 → entire score halved. depth_sim = 0.1 → score
collapses to near zero, regardless of 2D geometry. Objects at truly
different depths can never score HIGH.

Revisiting the failure case:
```
depth_sim = 0.5  (cat depth 0.9, vase depth 0.4)
2d_part   = 0.5 * 0.9 + 0.2 * 0.5 = 0.55
closeness = 0.55 * 0.5 = 0.275
risk_score = 0.275 * 1.5 = 0.41  → MEDIUM  ← correct
```

### Part 2 — Size-corrected depth proxy (improves depth signal quality)

Depth Anything produces a relative/ordinal closeness map. Two objects can
share the same depth value simply because they sit in the same depth band,
even if they are physically far apart. The pinhole camera model gives us:

```
pixel_size ∝ real_size_m / distance_m
⟹ distance_m ∝ real_size_m / pixel_size
```

A metric depth proxy computed from known object sizes and bounding-box
pixel area is independent of Depth Anything's internal scale. Blending
the two signals makes `depth_sim` more accurate for both failure modes.

```python
REAL_SIZE_M = {"cat": 0.35, "vase": 0.25, ...}  # rough medians in metres

def _size_depth_proxy(det, img_w, img_h) -> float | None:
    real_m = REAL_SIZE_M.get(det.class_name)
    if real_m is None:
        return None
    px_size = sqrt(px_w * px_h)              # bounding box area root
    proxy = real_m / px_size                 # large = far from camera
    return min(proxy / 0.05, 1.0)            # normalize; 0.05 = "very far"

def _depth_similarity_v2(a, b, img_w, img_h) -> float:
    da_sim    = 1.0 - abs(a.median_depth - b.median_depth)
    proxy_a   = _size_depth_proxy(a, img_w, img_h)
    proxy_b   = _size_depth_proxy(b, img_w, img_h)
    if proxy_a is None or proxy_b is None:
        return da_sim
    size_sim  = 1.0 - abs(proxy_a - proxy_b)
    return _ALPHA * size_sim + (1.0 - _ALPHA) * da_sim
```

**Alpha tuning note:** With depth now multiplicative (Part 1), proxy noise
has a stronger effect. Keep `_ALPHA` low (≤ 0.3). We use 0.25.

### Part 3 — Raise HIGH threshold

The multiplicative formula produces a tighter score range than the
original additive one. Raise the HIGH threshold from 0.6 → 0.65 to
avoid a new band of borderline false positives near the old threshold.

---

## Summary of Changes

| Change | File | Why |
|--------|------|-----|
| `closeness = (W1*prox + W3*contact) * depth` | mischief.py | depth becomes a gate, not a vote |
| `_depth_similarity_v2` blends DA + size proxy | mischief.py | corrects same-depth-band false matches |
| `_ALPHA = 0.25` | mischief.py | low blend weight — proxy has wide error bars |
| HIGH threshold `0.6 → 0.65` | mischief.py | compensates for tighter multiplicative range |

---

## Trade-offs

| Pro | Con |
|-----|-----|
| Depth now has veto power over 2D false positives | True positives where pet overlaps object at same depth are correctly preserved |
| No extra model or calibration required | Dog/cat size varies widely; proxy adds noise for unusual angles |
| `_ALPHA = 0.25` limits proxy noise exposure | Unknown-class objects fall back to raw DA signal only |
| Threshold bump reduces borderline false positives | May slightly increase false negatives near the HIGH/MEDIUM boundary |
