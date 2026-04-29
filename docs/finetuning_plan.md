# Fine-Tuning Plan: Improving Weak-Class Detection

## 1. Root-Cause Analysis

### 1.1 Per-Class Data Profile

| class | train imgs | test imgs | inst/img (train) | median bbox area | AP@0.5 |
|---|---|---|---|---|---|
| cat | 3,442 | 449 | 1.16 | **21.16%** | 0.8725 |
| dog | 3,677 | 446 | 1.26 | 10.69% | 0.8243 |
| laptop | 2,934 | 370 | 1.39 | 6.73% | 0.8015 |
| vase | 2,963 | 374 | 1.86 | 1.12% | 0.5811 |
| cup | 7,663 | 924 | **2.24** | 0.63% | 0.5768 |
| potted plant | 3,687 | 460 | 1.93 | 1.70% | 0.5165 |
| keyboard | **1,771** | 224 | 1.36 | 3.44% | 0.5719 |
| remote | 2,565 | 338 | 1.85 | **0.27%** | 0.4470 |

> Bbox area is expressed as a percentage of the full image (normalized w×h × 100).  
> At 640 px input: 0.27% area ≈ **21×21 px** object — near YOLO's effective detection floor.

---

### 1.2 Root Causes per Weak Class

#### `remote` — AP 0.4470 (worst)
- **Primary: extreme small-object size.** Median bbox = 0.27% → ~21 px at 640 resolution. Standard YOLO feature pyramids (P3/8, P4/16, P5/32) lose spatial detail for objects this small at the default stride.
- **Secondary: low training samples** (2,565 — third-fewest). At 1.85 instances/image, crowded scenes further confuse IoU matching.
- **Tertiary: high intra-class variance.** Remotes appear upright, sideways, partially occluded under couch cushions, etc.

#### `potted plant` — AP 0.5165
- **Primary: visual ambiguity.** Foliage lacks a crisp silhouette; bounding boxes often include non-plant background, making GT labels noisy. The model sees different leaf textures and structures as unrelated objects.
- **Secondary: small size** (1.70% median). Many background-merged instances are borderline detectable.
- **Secondary: high crowding** (1.93 inst/img) — multi-plant scenes produce overlapping boxes and hard-negative confusion.

#### `cup` — AP 0.5768 (worst AP despite most training data)
- **Sample count is NOT the issue** — cup has 7,663 training images, far more than cat (3,442). This rules out data imbalance as the primary explanation.
- **Primary: extreme small size** (0.63% median ≈ 50 px at 640). Many cups appear small in cluttered table scenes.
- **Primary: highest crowding** (2.24 inst/img). Cup scenes frequently have 3–10 instances. IoU-based NMS and AP matching both degrade in dense detection.
- **Secondary: massive intra-class shape variance.** Cups include mugs, espresso cups, paper cups, wine glasses, large bowls — a wider visual spread than any other class.

#### `vase` — AP 0.5811
- **Primary: small size** (1.12% median ≈ 68 px at 640).
- **Secondary: shape ambiguity.** Vases share silhouettes with bottles, jars, cups, and pots. The pretrained COCO model already conflates these.
- **Moderate sample size** (2,963 train) — not the bottleneck.

#### `keyboard` — AP 0.5719
- **Primary: lowest training sample count** (1,771 — 52% fewer than dog). This is the clearest case of data imbalance.
- **Secondary: extreme aspect ratio** (median 1.94 width/height — highly landscape). YOLO anchor priors are less tuned to wide flat objects.
- **Compensating factor: larger median area** (3.44%) — keyboards are physically large, which partly offsets the sample scarcity.

---

### 1.3 Summary Taxonomy

| cause | classes affected |
|---|---|
| Small object size (< 2% area) | remote, cup, vase, potted plant |
| High instance crowding (> 1.8 inst/img) | cup, potted plant, remote, vase |
| Low training sample count | keyboard, remote |
| Visual/shape ambiguity | cup, vase, potted plant |
| Aspect ratio mismatch | keyboard |

---

## 2. Fine-Tuning Strategy

Three orthogonal improvements, each targeting a distinct root cause.

### 2.1 Higher Input Resolution — targets small objects

The single highest-leverage fix for remote, cup, vase, and potted plant. At 640 px, a 0.27% bbox is ~21 px. At 1280 px it becomes ~43 px — 4× more pixels for the backbone to work with.

- Change `imgsz` from 640 → **1280**.
- YOLO's feature pyramid automatically gains an extra resolution level for small objects.
- Cost: inference time ~4× slower; train VRAM doubles. On Apple M3 Pro, prefer batch=4 with gradient accumulation (effective batch 16).
- Alternative at lower cost: **1024** — gives 2.56× area increase and fits with batch=8.

### 2.2 Copy-Paste Augmentation — targets keyboard and remote

Copy-paste pastes random object crops from other images into the training scene, effectively multiplying rare-class instances without collecting new data.

- Enable `copy_paste=0.5` in ultralytics training args.
- For maximum effect, combine with a **class-frequency-weighted sampler**: oversample images containing keyboard and remote so they appear proportionally to cup (7,663 images).
- Alternatively: create a **class-balanced dataset YAML** using `scripts/01_curate_coco.py` to re-curate with a cap (e.g., max 3,000 images per class) so the rarer classes are not diluted.

### 2.3 Scale and Density Augmentation — targets small + crowded objects

- **Aggressive scale jitter**: `scale=0.9` (current default is 0.5). This randomly crops into the scene and forces the model to detect objects at very small rendered sizes, simulating the remote/cup challenge.
- **Increased mosaic tile ratio**: keep `mosaic=1.0` but pair with `mixup=0.15` to blend two scenes — helps with multi-instance cup and potted plant scenes.
- **Erasing augmentation** (`erasing=0.4`): randomly blocks patches of the image, forcing the model to complete partial object detections — directly helps with occluded remotes.

### 2.4 Model Upgrade — increases capacity for ambiguous classes

`yolo11n` (nano, 2.6M params) struggles with high intra-class variance in cup and vase. Two options:

| option | params | pretrained mAP (COCO) | recommended for |
|---|---|---|---|
| `yolo11n` | 2.6M | ~39.5 | baseline, fast inference |
| `yolo11s` | 9.4M | ~47.0 | best accuracy/speed on M3 Pro |
| `yolo26s` | 9.5M | ~64.9 (our eval) | already available in repo |

`yolo26s` is already in the repo and shows strong pretrained baseline performance (mAP@0.5 = 0.649 without any fine-tuning). Fine-tuning it on our dataset is the highest-upside path. `yolo11s` is the standard alternative if a smaller fine-tuned model is preferred.

---

## 3. Training Configurations

### Baseline (current, for reference)
```python
PRETRAINED_WEIGHTS = "yolo11n.pt"
imgsz     = 640
epochs    = 100
batch     = 16
optimizer = "AdamW"
lr0       = 0.001
lrf       = 0.01
patience  = 20
mosaic    = 1.0
fliplr    = 0.5
hsv_h, hsv_s, hsv_v = 0.015, 0.7, 0.4
```

### Experiment A — Resolution bump (lowest risk, highest expected gain for small objects)
```python
PRETRAINED_WEIGHTS = "yolo11n.pt"   # same model, controls for architecture
imgsz     = 1280
batch     = 4
optimizer = "AdamW"
lr0       = 0.001
lrf       = 0.01
patience  = 20
# same augmentations as baseline
```
**Evaluation protocol for Exp A (see §4.1a):** evaluate at both imgsz=1280 and imgsz=640 to measure the resolution-mismatch penalty.

Expected: significant AP gain for remote, cup, vase; some gain for potted plant.

### Experiment B — Copy-paste + class balance (lowest risk for keyboard/remote)
```python
PRETRAINED_WEIGHTS = "yolo11n.pt"
imgsz     = 640
batch     = 16
copy_paste = 0.5       # key addition
mixup      = 0.15
erasing    = 0.4
scale      = 0.9
# class-balanced sampling via resampled dataset YAML
```
Expected: keyboard and remote AP gain from copy-paste; cup/potted plant from scale jitter.

### Experiment C — Model upgrade with fine-tuning (highest upside)
```python
PRETRAINED_WEIGHTS = "yolo26s.pt"   # already 0.649 mAP pretrained
imgsz     = 640                     # start at 640; can stack with Exp A
batch     = 8
optimizer = "AdamW"
lr0       = 0.0005     # lower LR: strong pretrained weights → gentler fine-tune
lrf       = 0.01
warmup_epochs = 5      # longer warmup for larger model
patience  = 25
copy_paste = 0.3
mixup      = 0.1
```
Expected: strongest overall AP, especially cup and vase due to greater model capacity.

### Experiment D — Targeted combination (data-driven, not all-at-once)

Exp D is constructed only after A, B, C complete. It combines **only the settings that showed positive, non-overlapping gains** in those runs. The exact config is therefore decided at evaluation time, not up front. See §4.2 for the decision tree.

Anticipated config (subject to revision):
```python
PRETRAINED_WEIGHTS = "yolo26s.pt"   # from C, if C beat B
imgsz     = ?          # from A/A′ cross-eval (see §4.1a)
copy_paste = ?         # retained only if B showed gain independent of C
scale      = 0.8       # retained if B showed gain
```
The `mixup` and `erasing` settings from Exp B are dropped from D unless B individually outperforms baseline without them — adding extra augmentations to an already-augmented run risks over-regularisation.

---

## 4. Experiment Design

### 4.1 Experiment Matrix

| exp | backbone | train imgsz | eval imgsz | copy_paste | purpose |
|---|---|---|---|---|---|
| Baseline | yolo11n | 640 | 640 | 0.0 | reference |
| A | yolo11n | 1280 | **1280** | 0.0 | resolution gain at matched eval |
| A′ | — | *(same weights as A)* | **640** | — | resolution-mismatch penalty |
| B | yolo11n | 640 | 640 | 0.5 | augmentation effect in isolation |
| C | yolo26s | 640 | 640 | 0.3 | model capacity effect in isolation |
| D | yolo26s | *decided after A/B/C* | *matched* | *decided* | targeted combination |

A and B train independently and can run in parallel. C depends only on the dataset. D is designed after evaluating A, B, C.

### 4.1a Resolution Cross-Evaluation (Concern 1)

Training at imgsz=1280 adapts the model's learned feature scales to high-resolution inputs. YOLO rescales inputs at inference time, but the internal anchor/stride calibration and the distribution of feature-map activations shift during training. Evaluating a 1280-trained model at 640 is therefore not neutral — it is a deliberate degradation test.

**Protocol for Exp A:**
1. Train once at imgsz=1280 → save `weights/best_1280.pt`.
2. Evaluate with `scripts/04_evaluate.py --weights best_1280.pt --imgsz 1280` → this is **Exp A** (the fair comparison for the resolution hypothesis).
3. Evaluate the *same weights* at `--imgsz 640` → this is **Exp A′** (the mismatch penalty).
4. Compare A, A′, and Baseline in a single table:

| exp | eval imgsz | remote AP | cup AP | overall mAP |
|---|---|---|---|---|
| Baseline | 640 | 0.447 | 0.577 | 0.649 |
| A | 1280 | ? | ? | ? |
| A′ | 640 | ? | ? | ? |

**Decision rule:**
- If A >> Baseline and A′ ≈ Baseline: resolution helps and the mismatch penalty is acceptable. Fix eval at 1280 going forward.
- If A >> Baseline but A′ << Baseline: resolution helps but the model is brittle to resolution change. Production use must specify imgsz=1280 explicitly. Carry imgsz=1280 into Exp D only if the deployment context permits it.
- If A ≈ Baseline: resolution is not the bottleneck; small objects may be too similar at both scales due to other factors (e.g., background complexity). Skip imgsz from Exp D.

This also guards against a common failure: a model that looks better at 1280 in the eval table but is slower in production without a matching improvement.

### 4.2 Combination Decision Tree (Concern 2)

Experiment D is built **bottom-up** from the results of A, B, C — not by blindly stacking all settings. Combining resolution, augmentation, and model size introduces three interaction risks:

| interaction | risk |
|---|---|
| High imgsz + copy-paste | Copy-paste crops are resized to imgsz; at 1280 the pasted objects may be rendered too large relative to scene, distorting the small-object distribution that made it useful |
| yolo26s + high imgsz | Memory-bound on M3 Pro (estimated batch=2 at 1280); effective batch may be too small for stable gradients |
| yolo26s + aggressive augmentation | Larger model can overfit augmented data if aug strength is tuned for nano; may need lower aug rates |

**Decision tree after evaluating A, B, C:**

```
Is A > Baseline by > 0.05 mAP?
├─ Yes → carry imgsz from A into D
│         Is A′ >> Baseline?  (mismatch penalty small?)
│         ├─ Yes → use train=1280, eval=1280 in D
│         └─ No  → use train=1280, eval=640 in D (accept penalty)
└─ No  → drop imgsz change; D uses imgsz=640

Is B > Baseline by > 0.03 mAP?
├─ Yes → carry copy_paste into D
│         Is C already using copy_paste?
│         ├─ Yes → lower copy_paste rate in D (0.3 → 0.2) to avoid stacking
│         └─ No  → keep at B's rate
└─ No  → drop copy_paste from D

Is C > Baseline by > 0.05 mAP?
├─ Yes → use yolo26s as D backbone
│         Check memory: if imgsz > 640 AND backbone=yolo26s → set batch >= 4
│         If batch < 4, reduce imgsz by one step (1280 → 1024 → 640)
└─ No  → keep yolo11n in D (no benefit from larger model for this dataset)
```

D is only run if at least two of A, B, C individually beat baseline — otherwise the most effective single change is the winner.

### 4.3 Metrics

Primary:
- `mAP@0.5` per class (weak 5 vs. strong 3)
- `mAP@[0.5:0.95]` overall

Secondary (diagnostic):
- Precision/Recall at conf=0.5 per weak class
- Training curve shape (overfit signal: val mAP plateau while train mAP climbs)

### 4.5 Evaluation Protocol

`scripts/04_evaluate.py` must accept an `--imgsz` argument (to be added before running Exp A) so each checkpoint can be evaluated at the correct resolution:

```bash
# Exp A: matched resolution
uv run scripts/04_evaluate.py --weights best_1280.pt --imgsz 1280

# Exp A′: mismatch penalty
uv run scripts/04_evaluate.py --weights best_1280.pt --imgsz 640

# Baseline and B/C: standard
uv run scripts/04_evaluate.py --weights best_640.pt --imgsz 640
```

All other aspects are unchanged: test labels are fixed (0–7 remapped IDs); COCO-pretrained models use the manual-inference path; fine-tuned checkpoints use `model.val()`.

### 4.6 Success Criteria

| class | baseline AP | target AP | minimum acceptable |
|---|---|---|---|
| remote | 0.4470 | > 0.60 | > 0.55 |
| potted plant | 0.5165 | > 0.65 | > 0.60 |
| cup | 0.5768 | > 0.70 | > 0.65 |
| vase | 0.5811 | > 0.70 | > 0.65 |
| keyboard | 0.5719 | > 0.70 | > 0.65 |
| cat | 0.8725 | maintain | > 0.85 |
| dog | 0.8243 | maintain | > 0.80 |
| laptop | 0.8015 | maintain | > 0.78 |

A run is rejected if any of the "maintain" classes drop below their minimum (regression guard).

### 4.7 Expected Outcome

| cause | fix | expected ∆AP |
|---|---|---|
| Small objects (remote, cup, vase) | imgsz 640→1280 | +0.08 – 0.15 |
| Low sample count (keyboard) | copy_paste=0.5 | +0.06 – 0.12 |
| Visual ambiguity (cup, vase) | yolo26s backbone | +0.05 – 0.10 |
| Crowding (cup, potted plant) | scale + mixup | +0.03 – 0.07 |
| Combined (Exp D) | all of the above | +0.12 – 0.20 overall |

---

## 5. Implementation Checklist (when ready to execute)

**Pre-requisites (script changes before any training):**
- [ ] Add `--imgsz` argument to `scripts/04_evaluate.py` so it can be passed to `model.val()` / `model.predict()`
- [ ] Add `--weights`, `--imgsz`, `--copy-paste`, `--backbone` CLI args to `scripts/03_train.py`
- [ ] Verify batch size headroom: run a short 1-epoch dry-run at imgsz=1280, batch=4 on M3 Pro before committing to full Exp A

**Phase 1 — parallel runs (A and B independent):**
- [ ] Run Exp A at imgsz=1280, ~2–3 h on M3 Pro
- [ ] Run Exp B at imgsz=640 with copy-paste, ~1.5 h on M3 Pro
- [ ] Evaluate Exp A at imgsz=1280 (Exp A) and imgsz=640 (Exp A′)
- [ ] Evaluate Exp B at imgsz=640
- [ ] Record mismatch penalty: ∆(A − A′); if > 0.05 mAP, flag for deployment note

**Phase 2 — sequential:**
- [ ] Run Exp C (yolo26s, imgsz=640), ~1.5 h
- [ ] Evaluate Exp C at imgsz=640
- [ ] Apply decision tree (§4.2) to determine Exp D config

**Phase 3 — combination (only if warranted):**
- [ ] Run Exp D with the data-driven config from §4.2
- [ ] Evaluate at the matched imgsz
- [ ] Compare all per-class AP tables; confirm no regression on cat/dog/laptop
- [ ] Declare winner; update `CLAUDE.md` with winning weights path, imgsz, and eval imgsz
