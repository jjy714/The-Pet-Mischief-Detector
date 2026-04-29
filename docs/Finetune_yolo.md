# YOLO Fine-Tuning Improvement Plan

## Step 0 — Diagnose before touching anything

Run `04_evaluate.py` and `01_curate_coco.py` (to get `class_distribution.csv`) if you haven't already. You need to know:

1. **Which classes have the lowest AP@0.5?** — those are your targets.
2. **What is the class image-count ratio?** — extreme imbalance (cup: 40k vs vase: 2k) drives you toward copy-paste augmentation and class weighting.
3. **Do the training curves show overfitting or underfitting?** — `model/runs/train/results.png`. Overfitting → more regularization. Underfitting → larger model or more epochs.

Everything in Steps 1–4 is conditional on what Step 0 reveals.

---

## Step 1 — Upgrade model: `yolo11n` → `yolo11s`

**Change:** `PRETRAINED_WEIGHTS = "yolo11s.pt"`

**Justification:**
- YOLOv11s has ~2× the parameters of v11n (~9M vs ~2.6M) and typically yields +3–6 mAP@0.5 on COCO-style tasks with no other changes.
- Since our 8 classes are a direct subset of COCO, the pretrained weights on `yolo11s.pt` already encode strong representations — fine-tuning converges fast despite the larger model.
- The inference speed penalty (~18ms vs ~10ms on GPU) is acceptable for this application.
- If GPU VRAM is tight at `batch=16`, drop to `batch=8`.

**Report note:** Compare `yolo11n` vs `yolo11s` mAP — the table quantifies the cost/benefit of model size.

---

## Step 2 — Add copy-paste and scale augmentation

**Changes to `model.train()`:**
```python
copy_paste = 0.3     # currently 0.0
mixup      = 0.1     # currently 0.0
scale      = 0.6     # currently 0.5 (Ultralytics default)
degrees    = 5.0     # currently 0.0
```

**Justification per parameter:**

| Param | Value | Why |
|---|---|---|
| `copy_paste` | 0.3 | Synthetically creates rare-class instances (vase, remote) by pasting them into other images. Direct fix for class imbalance without resampling. |
| `mixup` | 0.1 | Mild soft-label blending; acts as regularization to reduce overconfidence on common classes. |
| `scale` | 0.6 | Wider scale jitter (±60% vs ±50%) helps detect objects at varying distances — critical since the mischief pipeline needs to work at different room scales. |
| `degrees` | 5.0 | Slight rotation tolerance; objects on tables/shelves can be tilted in home environments. |

**Do not add:** `flipud=0.5` — household objects are never upside-down; this would degrade learning of object orientation cues.

---

## Step 3 — Two-phase freeze training

This is the right strategy specifically because all 8 of our classes are already in COCO pretrained weights — the backbone already has strong representations. We only need to adapt the detection head to our distribution.

**Phase 1 — Head warmup (epochs 1–30):**
```python
FREEZE_LAYERS = 10        # freeze all backbone layers
WARMUP_LR0    = 0.005     # higher LR since only head is updating
WARMUP_EPOCHS = 30
```

**Phase 2 — Full fine-tuning (epochs 1–100 from phase 1 best):**
```python
FREEZE_LAYERS = 0         # all layers trainable
LR0           = 0.0005    # lower LR to preserve backbone features
EPOCHS        = 100
```

**Why:** Starting with frozen backbone prevents the pretrained COCO backbone features from being distorted early in training when the head is still noisy. Phase 2 then fine-tunes end-to-end at a conservative LR.

**Implementation:** Call `model.train()` twice in sequence — phase 1 saves `model/runs/phase1/weights/best.pt`, phase 2 loads it and saves to `model/runs/train/weights/best.pt`. The script `03_train.py` should be updated to contain both phases as a single reproducible run.

---

## Step 4 — Hyperparameter adjustments

```python
LRF             = 0.005     # was 0.01 → deeper cosine decay (final LR = 0.0005 × 0.005 = 2.5e-6)
PATIENCE        = 25        # was 20 → more tolerance since phase 2 starts cold
LABEL_SMOOTHING = 0.1       # new — prevents overconfidence on common classes
WEIGHT_DECAY    = 0.0005    # explicitly set (matches Ultralytics default, document it)
```

**When to NOT touch:** `imgsz=640` — standard for YOLO; increasing to 1280 gives marginal gain but doubles VRAM and inference time. `optimizer=AdamW` — already optimal for fine-tuning pretrained models.

---

## Step 5 — Class imbalance mitigation (conditional on Step 0)

**If Step 0 shows ≥2 classes with AP@0.5 < 0.4:**

Option A — **Oversample rare classes during curation** (most robust): modify `01_curate_coco.py` to cap the common classes (cup, cat, dog) at 3× the count of the rarest class. Re-run scripts 01→02→03.

Option B — **`cls_pw` weighting**: pass a class-weight vector to upweight rare classes. In Ultralytics this is not a direct `model.train()` argument, but you can override the loss function. Document this in the report as a known limitation.

Option C — **Focal loss**: increase `fl_gamma` from 0.0 (default) to 1.5. Focal loss downweights easy examples (very common cup/cat detections) and forces the model to focus on hard/rare ones.

---

## Modified `03_train.py` structure

```python
# Phase 1 — backbone frozen, head only
model = YOLO("yolo11s.pt")
model.train(data=YAML, epochs=30, freeze=10, lr0=0.005, lrf=0.05,
            copy_paste=0.3, mixup=0.1, scale=0.6, degrees=5.0,
            label_smoothing=0.1, project=..., name="phase1")

# Phase 2 — full end-to-end from phase1 best
model = YOLO("<phase1>/weights/best.pt")
model.train(data=YAML, epochs=100, freeze=0, lr0=0.0005, lrf=0.005,
            copy_paste=0.3, mixup=0.1, scale=0.6, degrees=5.0,
            label_smoothing=0.1, patience=25, project=..., name="train")
```

---

## Report evidence to produce

Per the task requirement ("provide training logs and plots"):

1. **Training curves** — `model/runs/train/results.png` (Ultralytics auto-generates this): loss curves + mAP@0.5 vs epoch for both phases.
2. **Per-class AP table** — from `04_evaluate.py`: before vs after comparison.
3. **Hyperparameter table** — the table in `03_train.py`'s docstring already documents this; reference it in the report with justification per row.
4. **Class distribution plot** — from `01_curate_coco.py`'s `class_distribution.csv`: shows the imbalance that motivates copy-paste and focal loss choices.

---

## Priority order

If you only have time for some of these:

1. `yolo11s.pt` (Step 1) — biggest gain, one line change
2. `copy_paste=0.3` + `scale=0.6` (Step 2) — addresses imbalance and scale variance
3. Two-phase freeze (Step 3) — best justification for the report, moderate implementation work
4. Steps 4 and 5 — incremental tuning, do after seeing Step 0 results
