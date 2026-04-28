# Experiment Step V3 — Pipeline Evaluation & Depth Sampling Redesign

## History

| Version | Key change |
|---------|------------|
| v1 (Plan.md) | Original additive formula: `W1·prox + W2·depth + W3·contact` |
| v2 (depth_size_fix.md) | Multiplicative depth gate + size-range proxy with confidence-weighted alpha |
| **v3 (this doc)** | Full pipeline evaluation findings + portrait-line depth sampling proposal |

---

## Full Pipeline Evaluation Findings

### What the system actually detects

**The system is a proximity detector, not a mischief detector.**

A cat asleep next to a vase and a cat actively swatting at it produce identical
scores. The formula `(W1·prox + W3·contact) · depth` measures spatial cohabitation.
There is no motion, gaze, pose, or temporal signal anywhere in the pipeline. This
is acceptable for the assignment scope but must be stated clearly in the report.

---

### Critical issues

**C1 — Video startup false HIGH alerts** (`main.py:228-231`)

Until the depth thread produces its first map (~150 ms), every detection has
`median_depth = 0.0`. Then:

```
da_sim = 1.0 - |0.0 - 0.0| = 1.0  ← depth gate fully open
```

Every pair with 2D overlap fires HIGH for the first second of video.

**Fix**: gate output on `depth_count > 0`; emit a "LOADING" state until the
first real depth map arrives.

---

**C2 — Per-frame global depth normalization makes values frame-relative**
(`detector.py:114`)

```python
closeness = 1.0 - (depth - d_min) / (d_max - d_min)
```

The normalization is relative to whatever happens to be in the current frame.
Consequences:
- The same physical scene produces different `median_depth` values from frame to
  frame as the min/max shift with scene content.
- In flat scenes (everything at similar distance), noise is stretched to fill
  [0, 1] — two equidistant objects get opposite ends of the scale.
- Frame-to-frame values are not comparable → temporal reasoning is impossible.

This is the root cause the size-corrected proxy works around. A deeper fix would
maintain a running exponential average of per-frame min/max in video mode.

---

### Significant issues

**S1 — Bounding-box median depth is contaminated by background** (`detector.py:134`)

```python
roi = depth_map[y1:y2, x1:x2]
med = float(np.median(roi))
```

YOLO bboxes are loose. For a cat on a sofa, 30–50% of the bbox region is sofa
surface, not cat. The computed "cat depth" is a blend of cat and sofa, pulling it
toward the surface on which nearby objects also rest → false depth-match.

**Current partial fix**: inner 50% crop (v2 docs). A better approach is proposed
below (portrait line).

---

**S2 — `GAP_DECAY_PX = 50` is resolution-dependent** (`mischief.py:77`)

| Resolution | 50 px represents |
|------------|-----------------|
| 640 × 480  | ~10% frame width |
| 1920 × 1080 | ~2.6% frame width |
| 4K | ~1.3% frame width |

The same scene at different resolutions produces different `contact_likelihood`
scores. This breaks generalization across bonus test images and camera changes.

**Fix**: `GAP_DECAY = 0.05 × sqrt(img_w² + img_h²)` — consistent ~5% of the
image diagonal regardless of resolution.

---

**S3 — Unused `depth_map` parameter in `calculate_mischief`** (`mischief.py:188`)

The parameter is accepted but never read. Depth values are already embedded in
each `Detection.median_depth`. Should be removed from the signature.

---

**S4 — More objects in frame → higher false positive rate**

`max_risk = pairs[0].risk_score`. Six detected objects produce six pairs; each
is an independent chance for a 2D-overlap false positive. A messy room fires more
often than a sparse room, regardless of actual pet behavior.

**Fix**: temporal hysteresis — require HIGH to persist across N consecutive frames
before emitting the alert. Simultaneously fixes C1 (startup), S4 (multi-object
inflation), and the perpetual "cat sleeping by vase" alert.

---

### Moderate issues

**M1 — Thresholds and multipliers interact without design intent**

Single global threshold (0.65) + per-pair multipliers (1.0–1.5) create emergent
behavior:

| Pair | Multiplier | Closeness needed for HIGH |
|------|-----------|--------------------------|
| cat + vase | 1.5 | 0.43 |
| cat + cup | 1.5 | 0.43 |
| dog + laptop | 1.4 | 0.46 |
| dog + remote | 1.0 | **0.65** |

The asymmetry may be acceptable (fragile pairs deserve lower thresholds) but it
is an emergent side effect of two unrelated knobs, not a designed behavior.

---

**M2 — `_proximity_2d` has no size normalization**

Centroid distance ignores object size. A large cat whose edge is 2 px from a cup
can have the same centroid distance as a small cat 10 cm away. The `contact_likelihood`
edge bonus partially compensates, but the dominant W1 = 0.7 term does not.

---

## Portrait Line Proposal — Depth Sampling Redesign

### Problem with inner-crop

The v2 inner-crop (25% inset) improves over full-bbox but still samples a
rectangular 2D block. That block has:
- Horizontal background bleed: pixels to the left/right of the object silhouette
  are included, especially for objects with non-rectangular shapes (cats, plants)
- Symmetric treatment of depth direction: it does not distinguish the object's
  foreground surface from its background

### The portrait line concept

A **portrait line** is a narrow vertical strip at the center-x of the bounding
box, spanning the full (or near-full) height of the box:

```
cx = (x1 + x2) // 2
strip_w = max(4, (x2 - x1) // 5)   # ~20% of box width

sample region = depth_map[y1 : y2,  cx - strip_w//2 : cx + strip_w//2]
```

Visually:

```
┌──────────────────┐
│      bbox        │
│   ┌──────┐       │
│   │portrait│      │
│   │ strip │      │
│   └──────┘       │
└──────────────────┘
```

### Why it maximizes depth distance between objects

Objects in typical indoor scenes are arranged **horizontally** across the frame —
cups and vases sit to the side of a cat, not directly behind it. The portrait
strip eliminates horizontal background bleed almost entirely by sampling only
the center vertical column of the object. The only remaining contamination is
vertical: the floor below (for tall pets) and ceiling/wall above.

**Depth distance maximization argument:**

Consider a close cat (closeness ≈ 0.85) and a far vase (closeness ≈ 0.40).

| Sampling method | Cat sample | Vase sample | Difference |
|----------------|-----------|------------|------------|
| Full bbox | ~0.67 (blended with sofa) | ~0.48 (blended with shelf) | **0.19** |
| Inner 50% crop | ~0.75 (less sofa) | ~0.44 (less shelf) | **0.31** |
| Portrait strip | ~0.83 (mostly cat body) | ~0.41 (mostly vase body) | **0.42** |

The portrait strip produces the largest raw depth difference, giving the
multiplicative gate the strongest possible veto signal.

### Percentile vs. median

Using `np.median` gives the 50th percentile of the strip. For "maximizing
distance output", the **10th–25th percentile** (near-side) is better:

- For a close object: the near-side pixels are the object's front surface — the
  closest true reading. The median is dragged back by any background pixels.
- For a far object: near-side and median are nearly identical (the strip is
  homogeneously far), so the difference is negligible.

Net effect: **close objects get closer, far objects stay far → larger depth
difference → stronger gate**.

```python
# Near-side percentile sampling on the portrait strip
pixels = depth_map[iy1:iy2, ix1:ix2].ravel()
med = float(np.percentile(pixels, 25)) if pixels.size > 0 else 0.0
```

### Portrait line parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Strip width | `max(4, (x2-x1) // 5)` | ~20% of box width; min 4 px to avoid noise |
| Vertical span | `y1 + bh//8` to `y2` | Drop top 12.5% (hat/ceiling artifacts) |
| Sampling | 25th percentile | Near-side bias; maximizes close/far separation |
| Fallback | Full bbox median | When box is < 20 px wide (small objects at distance) |

### Comparison: inner-crop vs. portrait line

| Property | Inner 50% crop | Portrait line |
|----------|---------------|---------------|
| Horizontal bleed | Reduced by 50% | Eliminated (~0%) |
| Vertical bleed | Reduced by 50% | Minimal (floor only) |
| Noise robustness | High (large area) | Moderate (narrow strip) |
| Typical sample area | 25% of bbox | ~10% of bbox |
| Fallback needed? | Only for degenerate boxes | Yes, for narrow boxes |
| Depth distance output | Moderate improvement | Maximum improvement |

**Verdict**: portrait line + 25th percentile is strictly better for depth
discrimination. The tradeoff is that it requires a minimum box width fallback
for very small detections (remote at long range, cup partially in frame).

---

## Implementation Plan (v3)

Priority order:

| # | Change | File | Lines affected |
|---|--------|------|---------------|
| 1 | Portrait line + 25th-percentile sampling in `fill_depths` | `detector.py` | `fill_depths()` |
| 2 | Resolution-relative `GAP_DECAY` | `mischief.py` | constant + `_contact_likelihood` |
| 3 | Startup gate on `depth_count > 0` | `main.py` | `run_video()` |
| 4 | Remove unused `depth_map` param from `calculate_mischief` | `mischief.py` + callers | signature |
| 5 | (Optional) Temporal hysteresis — N-frame HIGH requirement | `main.py` | `run_video()` |

---

## Expected Impact

After all v3 changes:

```
Old fill_depths (full bbox):
  cat median_depth ≈ 0.67  (contaminated by sofa)
  vase median_depth ≈ 0.48 (contaminated by shelf)
  depth_sim = 1 - |0.67 - 0.48| = 0.81  ← weak gate

New fill_depths (portrait + 25th pct):
  cat median_depth ≈ 0.83  (cat's front surface)
  vase median_depth ≈ 0.41 (vase's front surface)
  depth_sim = 1 - |0.83 - 0.41| = 0.58  ← stronger gate
  closeness = (0.7·0.9 + 0.3·0.5) · 0.58 = 0.45
  risk_score = 0.45 · 1.5 = 0.67  → borderline HIGH
  (with temporal hysteresis: held for N frames before emitting)
```

The portrait line alone reduces false-positive rate substantially. Combined with
temporal hysteresis, sustained false positives (cat sleeping by vase) become
addressable without touching the threshold constants.
