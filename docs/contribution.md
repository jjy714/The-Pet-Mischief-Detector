# Personal Contributions (HEO Sunghak)

This document summarizes my individual contributions to the Pet Mischief Detector project, covering system design, experimentation, debugging, and analysis.

---

## 1. Identifying the 2D Spatial Limitation and Proposing the Depth Pipeline

The first critical design decision I made was recognizing that a purely 2D bounding box approach is fundamentally insufficient for spatial proximity reasoning.

Objects that appear close in a 2D image frame may be at entirely different depths in 3D space — a cat and a vase on separate shelves can appear adjacent in the image even though they cannot physically interact. Conversely, two objects far apart in the image may actually be at the same depth plane and physically close.

To demonstrate this concretely, I created reference images in `imgs/`:

- `two_objects_low_depth_close_eachother.jpeg` — objects near the camera appear far apart in 2D
- `two_objects_higher_depth_close_eachother.jpeg` — objects far from the camera appear close in 2D

In both images, the objects occupy the same relative 3D positions; only camera distance differs. This illustrates that 2D proximity alone is a misleading signal.

Based on this analysis, I proposed a two-model pipeline:

- **YOLO** for 2D object detection
- **Depth Anything V2** for monocular depth estimation

The depth map is used to approximate 3D spatial relationships, acting as a multiplicative gate on the 2D proximity score. This prevents false positives when objects are on different depth planes. I designed the overall pipeline architecture; the full implementation was carried out by Jason.

---

## 2. End-to-End Pipeline Execution and Validation

I executed the complete inference pipeline — YOLO detection, depth inference, mischief scoring, and visualization — and validated that annotated outputs were generated correctly under `outputs/visualizations_*`.

This ensured that the system ran end-to-end before any quantitative evaluation, allowing us to catch integration issues early.

---

## 3. Evaluation Pipeline Construction and Debugging

### 3.1 Class Index Mismatch (Pretrained vs. Custom Classes)

Running `scripts/04_evaluate.py` against the pretrained `yolo26s.pt` (80 COCO classes) on our custom 8-class test set produced mAP ≈ 0. I diagnosed this as a **class index mismatch**: YOLO evaluates by class index, but our 8-class IDs (0–7) do not correspond to the COCO class IDs for the same objects (e.g., our class 0 = `cat` maps to COCO ID 15).

I resolved this by using `scripts/06_eval_risk_labels.py`, which matches predictions by class name rather than index, producing valid evaluation results for pretrained models.

### 3.2 Dataset Labeling Inconsistencies

During evaluation, I identified structural problems in the hand-labelled risk test set:

- Multiple risk labels assigned per image
- Risk labels assigned to non-pet objects rather than to the scene as a whole
- Ambiguity between object-level and image-level interpretation

Since our system outputs a single image-level risk label, these inconsistencies made evaluation unreliable. I raised the issue and the labelling convention was revised to **one risk label per image**, making evaluation consistent and meaningful.

---

## 4. YOLO Baseline Experiments (Pretrained Model Comparison)

### 4.1 Risk-Label Accuracy: yolo11n vs. yolo26s

Using the corrected evaluation pipeline, I ran risk-label classification experiments comparing the two pretrained baselines on the same test set.

**Before logic fix:**

| Model | Accuracy |
|---|---|
| yolo11n + Depth Anything V2 | 0.2908 |
| yolo26s + Depth Anything V2 | 0.2482 |

Confusion matrix — yolo11n:

```
GT \ Pred   LOW   MEDIUM   HIGH
LOW          15        2     11
MEDIUM       19        7     61
HIGH          6        1     19
```

Confusion matrix — yolo26s:

```
GT \ Pred   LOW   MEDIUM   HIGH
LOW          10        1     17
MEDIUM       12        6     69
HIGH          4        3     19
```

**Key observation:** Both models severely over-predicted HIGH risk. Depth information was present but not influential enough in the scoring formula, and the larger model (26s) did not outperform the smaller one under the flawed logic.

### 4.2 Impact of Risk Scoring Logic Update

After the mischief scoring logic was updated by Jason (depth as multiplicative gate, edge-gap proximity, IoU-based contact likelihood), I re-ran all baseline experiments to measure the impact.

**yolo11n + Updated Logic**

- Accuracy: **0.5674**

```
GT \ Pred   LOW   MEDIUM   HIGH
LOW          23       14      7
MEDIUM        0        0      2
HIGH         23       15     57
```

**yolo26s + Updated Logic**

- Accuracy: **0.5390**

```
GT \ Pred   LOW   MEDIUM   HIGH
LOW          17       20      7
MEDIUM        0        0      2
HIGH         15       21     59
```

Accuracy nearly doubled relative to the original scoring logic. The main improvement was a drastic reduction in spurious HIGH predictions — a direct result of depth becoming a multiplicative gate rather than an additive term. When objects are at different depth planes, the score collapses to near zero regardless of their 2D proximity.

Notably, model size (11n vs. 26s) had less impact on risk accuracy than the scoring logic design. This confirms that the system's bottleneck was the mischief scoring formula, not the detection backbone.

### 4.3 Official YOLO mAP Evaluation (yolo26s Baseline)

I also ran a standard YOLO mAP evaluation using `scripts/04_evaluate.py` on the pretrained `yolo26s.pt` with class-name remapping, establishing an upper-bound object detection baseline.

| Metric | Value |
|---|---|
| mAP@0.5 | **0.6489** |
| mAP@0.5–0.95 | **0.5286** |

Per-class AP@0.5:

| Class | AP@0.5 |
|---|---|
| cat | 0.8725 |
| dog | 0.8243 |
| laptop | 0.8015 |
| keyboard | 0.5719 |
| vase | 0.5811 |
| cup | 0.5768 |
| potted plant | 0.5165 |
| remote | 0.4470 |

This confirms that the pretrained model already performs well on common classes (cat, dog, laptop) while struggling with small objects (remote, cup).

---

## 5. Fine-Tuning Analysis and Experiments

### 5.1 Per-Class Root Cause Analysis and Fine-Tuning Plan

I wrote a comprehensive root-cause analysis of per-class detection weaknesses in `docs/finetuning_plan.md`. For each underperforming class, I identified the primary failure mode:

| Class | AP@0.5 | Primary Issue |
|---|---|---|
| remote | 0.4470 | Extreme small size (~21 px at 640 input); high crowding |
| potted plant | 0.5165 | Visual ambiguity (irregular silhouette); small size |
| cup | 0.5768 | Small size (0.63% bbox area); high crowding; shape variance |
| vase | 0.5811 | Small size; shape overlap with bottles and jars |

Classes with large median bbox area (cat, dog) and adequate data performed well. The analysis demonstrated that small object size and crowding were the dominant bottlenecks, not sample count per se (cup has 7,663 training images yet AP = 0.58).

### 5.2 Script Extension: `--fraction` and `--freeze` Arguments

I extended `scripts/03_train.py` with two new CLI arguments to support controlled fine-tuning experiments:

- `--fraction` — uses a reproducible subset of training data (e.g., `0.1` → 2,201 images), enabling faster iteration without resampling
- `--freeze N` — freezes the first N backbone layers during training, allowing the detection head to be trained independently at a higher learning rate

These arguments were passed directly to Ultralytics `model.train()` and made all subsequent fine-tuning experiments reproducible and configurable.

### 5.3 Exp C: Initial Fine-Tuning Attempt (mAP = 0 Finding)

I ran the initial fine-tuning experiment (`exp_c`) over 30 epochs:

```
yolo26s.pt  →  8-class fine-tuning
fraction=0.1, lr0=0.0005, batch=8, epochs=30
```

Despite validation losses decreasing steadily (indicating the backbone was learning), **mAP@0.5 remained 0.0 throughout all 30 epochs**. A formal evaluation of the saved `best.pt` confirmed:

```
mAP@0.5      = 0.0
mAP@0.5–0.95 = 0.0
precision    = 0.0002
recall       = 0.0001
```

### 5.4 Root Cause Diagnosis: Learning Rate Too Low for Reinitialized Head

I identified the cause of the mAP=0 failure: when fine-tuning a pretrained YOLO model from 80 classes to 8 classes, **the detection head is randomly reinitialized** because the output dimension changes. This new head requires a much higher learning rate to learn meaningful predictions.

With `lr0=0.0005`, the backbone updated (validation loss decreased), but the randomly initialized head never calibrated to produce confident, class-specific predictions. Inference confirmed this: all detections were below 2% confidence.

The fix was to use a two-phase training strategy: first freeze the backbone and train only the head at a higher learning rate, then optionally unfreeze the entire network at a lower rate.

### 5.5 Apple Silicon MPS Bug: Discovery and Patch

During training, the process repeatedly crashed with:

```
RuntimeError: zeros: Dimension size must be non-negative
```

I traced this to a bug in `ultralytics/utils/loss.py`: on Apple Silicon MPS devices, `batch_idx.unique(return_counts=True)` returns garbage (including negative) count values, causing `torch.zeros(batch_size, counts.max(), ...)` to fail.

I patched two occurrences (standard Detect head, line ~377; OBB head, line ~998) by moving the tensor to CPU before reading the max:

```python
# Before (crashes on MPS):
out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)

# After (correct):
max_count = max(int(counts.cpu().max().item()) if counts.numel() > 0 else 0, 0)
out = torch.zeros(batch_size, max_count, ne - 1, device=self.device)
```

This fix resolved all crashes and allowed training to proceed without interruption on MPS.

### 5.6 Exp C Freeze: Two-Phase Training Strategy

Based on the root cause diagnosis, I designed and launched a corrected fine-tuning experiment (`exp_c_freeze`):

- **Phase 1 (freeze=10):** freeze the first 10 backbone layers; train only the neck and detection head
- **Learning rate:** `lr0=0.005` — approximately 10× higher than the original, appropriate for a randomly initialized head
- **Warmup:** 3 epochs; patience=10; 20 epochs total; `fraction=0.1`

This approach matches the two-phase strategy described in the fine-tuning literature: stabilize the new head before allowing gradient flow into the pretrained backbone.

**Results:** *(to be filled in once training completes)*

| Metric | exp_c (failed) | exp_c_freeze |
|---|---|---|
| mAP@0.5 | 0.000 | TBD |
| mAP@0.5–0.95 | 0.000 | TBD |

---

## Summary of Contributions

| Area | Contribution |
|---|---|
| System design | Identified 2D limitation; proposed YOLO + Depth Anything V2 pipeline |
| Evaluation infrastructure | Diagnosed class-index mismatch; built class-name-based evaluation; fixed label inconsistencies |
| Baseline experiments | Compared yolo11n vs. yolo26s on risk-label accuracy; measured pretrained mAP (0.6489) |
| Fine-tuning analysis | Wrote per-class root-cause analysis; identified small-object and crowding bottlenecks |
| Script development | Added `--fraction` and `--freeze` to `scripts/03_train.py` |
| Fine-tuning experiments | Ran Exp C (30 epochs); diagnosed mAP=0 as LR-too-low for reinitialized head |
| Bug fix | Discovered and patched MPS-specific crash in `ultralytics/utils/loss.py` |
| Two-phase training | Designed and launched `exp_c_freeze` (freeze=10, lr=0.005) to correct the training strategy |
| Results analysis | Quantified impact of scoring logic changes; interpreted confusion matrices |
