# COMP4423 Computer Vision — Project Report  
## The Pet Mischief Detector

**Student Name:** HEO Sunghak  
**Student ID:** 21097305d  
**Group Members:** HEO Sunghak (21097305d), Jason Jung, Jason Jeong  
**Submission Date:** April 30, 2026  

---

## Abstract

This report documents the Pet Mischief Detector, a computer vision system that classifies the risk level of pet–object interactions (LOW / MEDIUM / HIGH) in real time. The system fuses YOLO-based object detection with monocular depth estimation (Depth Anything V2) to reason about 3D spatial proximity. This report focuses on my individual contributions: proposing the depth-aware pipeline architecture to overcome the limitations of 2D proximity reasoning; constructing and debugging the evaluation infrastructure; running quantitative baseline experiments comparing two YOLO backbones; conducting per-class detection analysis; and designing and executing fine-tuning experiments including diagnosing a learning rate failure mode and patching an Apple Silicon MPS crash in the Ultralytics library. Fine-tuning results from the corrected two-phase training run are pending at submission time.

---

## Statement of Contribution

| Task | My Contribution |
|---|---|
| Task 1: Problem Definition & Data Curation | Participated in class selection; identified the 2D spatial limitation that motivated the depth pipeline; created reference images demonstrating the 2D-vs-3D distortion |
| Task 2: Dataset Splitting | Executed and validated the train/val/test split pipeline |
| Task 3: Model Training | Wrote and ran all fine-tuning experiments; extended `scripts/03_train.py` with `--fraction` and `--freeze`; diagnosed mAP=0 failure; patched Apple Silicon MPS crash; designed two-phase training strategy; wrote `docs/finetuning_plan.md` |
| Task 4 Part A: Quantitative Evaluation | Ran all mAP and risk-label evaluations using `scripts/04_evaluate.py` and `scripts/06_eval_risk_labels.py`; diagnosed and fixed class-index mismatch |
| Task 4 Part B: System Evaluation | Ran end-to-end pipeline; produced annotated visualizations; identified and fixed dataset labeling inconsistencies; analyzed before/after impact of scoring logic changes |

Jason Jung and Jason Jeong were primarily responsible for: the mischief scoring logic (`model/mischief.py`) and its iterative improvements; the data curation and splitting scripts; and the video-mode two-thread architecture.

---

## 1. Task 1: Problem Definition and Data Curation

### 1.1 Class Selection

The system detects 8 classes: **cat**, **dog** (pets) and **cup**, **laptop**, **potted plant**, **vase**, **remote**, **keyboard** (household objects that pets can knock over, chew, or contaminate). This selection targets the most common categories of pet mischief in indoor environments and is drawn entirely from the COCO 2017 dataset, avoiding the need for manual annotation.

### 1.2 The 2D Spatial Limitation — Motivating the Depth Pipeline

The most fundamental design decision in the project was recognizing that 2D bounding box proximity is insufficient for spatial interaction reasoning. Consider two cases:

- A cat and a cup on **separate shelves** appear side-by-side in the image — bounding boxes nearly overlap — yet they are separated by half a metre of vertical space and cannot interact.
- A cat and a laptop on the **same table** appear far apart in a telephoto image, yet their physical distance is negligible.

2D proximity alone cannot distinguish these cases. I created two reference images in `imgs/` to demonstrate this:

| Image | What it shows |
|---|---|
| `two_objects_low_depth_close_eachother.jpeg` | Near-camera objects appear far apart in 2D despite close 3D proximity |
| `two_objects_higher_depth_close_eachother.jpeg` | Far-camera objects appear close in 2D despite a different depth plane |

> **Figure 1** — Reference images demonstrating 2D-vs-3D projection distortion.  
> See `imgs/two_objects_low_depth_close_eachother.jpeg` and `imgs/two_objects_higher_depth_close_eachother.jpeg`.

Based on this analysis, I proposed fusing YOLO object detection with Depth Anything V2 monocular depth estimation, using depth as a **multiplicative gate** on the 2D proximity score. This prevents false positives when objects appear close in 2D but are at different depth planes.

### 1.3 Dataset Curation

The dataset was assembled by filtering COCO 2017 (`scripts/01_curate_coco.py`) to retain only images containing at least one of the 8 target classes. After filtering:

| Split | Images |
|---|---|
| Training | ~22,000 |
| Validation | ~2,751 |
| Test (COCO-sourced) | ~2,754 |

In addition, a hand-labelled **risk-label test set** of 141 images was assembled for evaluating mischief classification performance, with one ground-truth risk label (LOW / MEDIUM / HIGH) per image.

---

## 2. Task 2: Dataset Splitting and Formatting

The curated dataset was split into train, validation, and test sets using `scripts/02_split_dataset.py`. A fixed random seed ensures reproducibility. Data is formatted in **YOLO TXT format** — one `.txt` file per image containing normalized `[class_id, cx, cy, w, h]` bounding box annotations — as required by the Ultralytics training framework. A `schema/dataset.yaml` configuration file points to the split directories and maps class indices to names.

The `fraction` parameter in `scripts/03_train.py` (added by me, described in Section 4) allows training on a reproducible subset without re-running the split.

---

## 3. System Architecture

The detector fuses two models in tandem:

- **YOLO** (`yolo11n.pt` or `yolo26s.pt`) — detects bounding boxes and class labels for 8 target classes
- **Depth Anything V2 Small** — produces a monocular closeness map in [0, 1] (1 = near camera, 0 = far)

```
frame
  ├─→ YOLO                  →  [Detection, ...]       (bounding boxes + classes)
  └─→ Depth Anything V2     →  depth_map              (H×W, [0,1])
                                     ↓
                               fill_depths()           (sample depth per box)
                                     ↓
                           calculate_mischief()        (score every pet–object pair)
                                     ↓
                             draw_frame()              (annotate frame, show risk)
```

For each (pet, object) pair, the mischief score is:

```
closeness  = (W1 · proximity_2d + W3 · contact_likelihood) × depth_similarity
risk_score = closeness × pair_multiplier
```

The depth signal acts as a **multiplicative gate**: when two objects have dissimilar depth values (different planes), `depth_similarity` collapses, suppressing the 2D proximity score to near zero. Thresholds: `risk_score > 0.65` → HIGH, `> 0.3` → MEDIUM, else LOW.

This architecture was proposed by me and implemented by Jason Jung/Jeong.

---

## 4. Task 3: Model Training and Fine-Tuning

### 4.1 Model Selection Rationale

Two pretrained YOLO backbones were evaluated:

- **yolo11n.pt** — lightweight (nano), fast inference on CPU/MPS, low memory footprint
- **yolo26s.pt** — larger (small), approximately 2× more parameters, better capacity for small-object detection

Both are pretrained on COCO 2017 (80 classes). Fine-tuning adapts these to our 8-class subset. The `yolo26s` variant was selected as the primary fine-tuning target because, as the per-class analysis (Section 4.3) shows, the weakest classes (remote, cup, vase) suffer from small object size and high crowding — problems that benefit from additional backbone capacity.

### 4.2 Script Extension: `--fraction` and `--freeze`

I extended `scripts/03_train.py` with two arguments to support controlled experiments:

```python
p.add_argument("--fraction", type=float, default=1.0,
               help="Fraction of training data (0.0–1.0); 0.1 → 2,201 images")
p.add_argument("--freeze",   type=int,   default=0,
               help="Freeze first N backbone layers (0 = none)")
```

- `--fraction 0.1` selects a reproducible 10% subset (2,201 images) for rapid iteration — approximately 10× faster per epoch than the full training set, at the cost of reduced data diversity.
- `--freeze N` freezes backbone layers 0 through N−1, concentrating gradient updates on the neck and detection head. This is critical when the detection head is randomly reinitialized (see Section 4.4).

### 4.3 Per-Class Detection Analysis

Before fine-tuning, I established a pretrained baseline and performed a root-cause analysis of per-class weaknesses (`docs/finetuning_plan.md`).

**Table 1 — Pretrained yolo26s.pt: detection performance on 8-class test set**

| Metric | Value |
|---|---|
| mAP@0.5 | **0.6489** |
| mAP@0.5:0.95 | **0.5286** |

**Table 2 — Per-class AP@0.5 and object size profile**

| Class | AP@0.5 | Train images | inst/img | Median bbox area |
|---|---|---|---|---|
| cat | 0.8725 | 3,442 | 1.16 | 21.16% |
| dog | 0.8243 | 3,677 | 1.26 | 10.69% |
| laptop | 0.8015 | 2,934 | 1.39 | 6.73% |
| keyboard | 0.5719 | 1,771 | 1.36 | 3.44% |
| vase | 0.5811 | 2,963 | 1.86 | 1.12% |
| cup | 0.5768 | 7,663 | 2.24 | 0.63% |
| potted plant | 0.5165 | 3,687 | 1.93 | 1.70% |
| remote | 0.4470 | 2,565 | 1.85 | 0.27% |

> **Figure 2** — Per-class AP@0.5 bar chart.  
> See `outputs/eval/per_class_ap.png`

**Root cause analysis:**

| Failure mode | Affected classes |
|---|---|
| Small object size (bbox < 2% area at 640px) | remote (~21px), cup (~50px), vase (~68px), potted plant |
| High instance crowding (> 1.8 inst/img) | cup (2.24), potted plant (1.93), vase (1.86), remote (1.85) |
| Low training sample count | keyboard (1,771 — fewest), remote |
| Shape/visual ambiguity | cup, vase, potted plant |

The `remote` class is the worst case: median bbox area 0.27% corresponds to ~21×21 pixels at 640px input — near YOLO's effective detection floor (P3/8 stride). Crucially, `cup` has the most training images (7,663) yet achieves only AP 0.58, demonstrating that **sample count is not the primary bottleneck** — small object size and crowding are.

### 4.4 Experiment C: Initial Fine-Tuning (mAP = 0 Failure)

**Configuration:**

| Parameter | Value |
|---|---|
| Backbone | yolo26s.pt |
| Optimizer | AdamW |
| Learning rate (lr0) | 0.0005 |
| Batch size | 8 |
| Epochs | 30 |
| fraction | 0.1 (2,201 images) |
| freeze | None (full network) |
| Device | Apple Silicon MPS |

**Table 3 — Exp C training metrics (selected epochs)**

| Epoch | train/box_loss | train/cls_loss | val/box_loss | val/cls_loss | mAP@0.5 |
|---|---|---|---|---|---|
| 1 | 3.337 | 5.747 | 3.340 | 5.541 | 0.000 |
| 5 | 3.295 | 5.362 | 3.204 | 5.552 | 0.000 |
| 30 | — | — | — | — | 0.000 |

Despite 30 epochs of training, **mAP@0.5 remained 0.0 throughout**. Formal evaluation of the saved `best.pt`:

```
mAP@0.5      = 0.000
mAP@0.5:0.95 = 0.000
precision    = 0.0002
recall       = 0.0001
```

Manual inference confirmed predictions existed but at < 2% confidence — well below any usable detection threshold.

**Root cause — learning rate too low for randomly reinitialized head:**

When fine-tuning from 80 classes to 8 classes, the YOLO detection head is **randomly reinitialized** because the output dimension changes (80 × regression+classification → 8 ×). This creates a fundamental conflict:

- The pretrained backbone requires a **low learning rate** (< 0.001) to avoid catastrophic forgetting.
- The randomly initialized head requires a **high learning rate** (≥ 0.005) to calibrate in a reasonable number of epochs.

With `lr0=0.0005` applied uniformly, the backbone learned (validation loss decreased) while the head was chronically under-trained. The decreasing validation loss was therefore a misleading signal — it reflected backbone feature improvement, not head calibration.

### 4.5 Apple Silicon MPS Bug: Discovery and Patch

During training, the process crashed repeatedly at epochs 13, 20, and 27:

```
RuntimeError: zeros: Dimension size must be non-negative
  File ".../ultralytics/utils/loss.py", line 377
    out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
```

I initially suspected the `copy_paste` augmentation creating empty target tensors. Disabling `copy_paste` did not fix the crash — ruling out that hypothesis.

I traced the root cause to an **Apple Silicon MPS-specific PyTorch bug**: `batch_idx.unique(return_counts=True)` returns garbage (including negative) values on MPS under certain batch conditions. The subsequent `.max()` call produces a negative integer, and `torch.zeros(batch_size, <negative>, ...)` raises the exception.

I patched two locations in `ultralytics/utils/loss.py` — the standard Detect head (~line 377) and the OBB head (~line 998):

```python
# Before (crashes on MPS):
out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)

# After (correct on all devices):
max_count = max(int(counts.cpu().max().item()) if counts.numel() > 0 else 0, 0)
out = torch.zeros(batch_size, max_count, ne - 1, device=self.device)
```

Moving the tensor to CPU before reading (`.cpu()`) avoids the MPS numerical error. The outer `max(..., 0)` clamps any residual negative to zero. This fix resolved all crashes and allowed training to proceed uninterrupted.

### 4.6 Experiment C Freeze: Two-Phase Training Strategy

Based on Section 4.4, I designed a corrected experiment (`exp_c_freeze`):

**Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| lr0 | **0.005** | 10× higher than Exp C; appropriate for randomly initialized head |
| freeze | **10** | Backbone layers 0–9 frozen; head and neck trained freely |
| warmup_epochs | 3 | Gradual LR ramp to prevent initial instability |
| patience | 10 | Early stopping if no improvement for 10 epochs |
| epochs | 20 | Sufficient for head calibration |
| fraction | 0.1 | Same 2,201 images as Exp C for fair comparison |

Freezing the backbone prevents pretrained weights from being disrupted during head calibration while concentrating the full learning rate on the neck and detection head. This two-phase approach — stabilize head first, optionally unfreeze backbone later — is the standard strategy for class-count changes during fine-tuning.

**Table 4 — Exp C Freeze training results (in progress)**

| Epoch | train/box_loss | train/cls_loss | val/box_loss | val/cls_loss | mAP@0.5 |
|---|---|---|---|---|---|
| 1 | 3.349 | 5.720 | 3.234 | 5.576 | 0.000 |
| 2 | 3.355 | 5.422 | 3.221 | 5.495 | 0.000 |
| … | … | … | … | … | *(training in progress)* |

> **Note:** Training is ongoing at submission time. Table 5 will be updated when training completes.

**Table 5 — Fine-tuning experiment summary** *(to be completed)*

| Experiment | lr0 | freeze | Epochs | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| Pretrained yolo26s (no fine-tuning) | — | — | — | 0.6489 | 0.5286 |
| Exp C | 0.0005 | None | 30 | 0.000 | 0.000 |
| Exp C Freeze (phase 1) | 0.005 | 10 | 20 | *TBD* | *TBD* |

**Justification of training choices:**
- **AdamW optimizer**: more stable than SGD for fine-tuning pretrained models, especially with sparse gradients from small-object classes.
- **fraction=0.1**: reduces epoch time from ~14 min (full set) to ~14 min — the training set itself is a 10% fraction at 2,201 images, acceptable given the backbone retains strong COCO priors.
- **Mosaic augmentation (1.0) + fliplr (0.5)**: standard YOLO augmentations retained for context diversity. `flipud=0.0` to avoid unnatural orientations for indoor scenes.

---

## 5. Task 4: Mischief System Evaluation

### 5.1 Part A: Quantitative Object Detection Evaluation

**Metrics:** mAP@0.5 and mAP@0.5:0.95, the standard metrics for object detection benchmarking. mAP@0.5 measures detection quality at a single IoU threshold (0.5) and is widely used for comparison. mAP@0.5:0.95 averages over IoU thresholds 0.5–0.95 at 0.05 steps and provides a more demanding measure of localization precision.

**Evaluation method:** I used `scripts/04_evaluate.py` for the fine-tuned model and a class-name-remapped version for the pretrained baseline. A critical debugging step was identifying a **class-index mismatch**: the pretrained `yolo26s.pt` assigns class index 15 to `cat`, 16 to `dog`, etc. (COCO indices), while our dataset reindexes these as 0–7. Evaluating by index would produce mAP=0 for a visually correct model. I resolved this by using `scripts/06_eval_risk_labels.py`, which evaluates by class name.

**Results:** See Tables 1 and 2 in Section 4.3.

### 5.2 Part B: Qualitative System Evaluation

#### Risk-Label Accuracy Experiments

To evaluate the full mischief system, I ran the complete pipeline on the 141-image risk-label test set using both YOLO backbones. The test set distribution:

| Risk Level | Count | Proportion |
|---|---|---|
| HIGH | 95 | 67.4% |
| LOW | 44 | 31.2% |
| MEDIUM | 2 | 1.4% |

**Dataset labeling fix:** Before evaluation could be trusted, I identified structural inconsistencies in the test labels — multiple labels per image, and labels assigned to individual objects rather than the scene. I raised this issue and the convention was standardized to **one image-level risk label per image**.

**Table 6 — Risk-label accuracy: before vs. after scoring logic update**

| Model | Before logic fix | After logic fix | Improvement |
|---|---|---|---|
| yolo11n + Depth Anything V2 | 0.2908 | **0.5674** | +0.277 |
| yolo26s + Depth Anything V2 | 0.2482 | **0.5390** | +0.291 |

**Table 7 — Confusion matrix: yolo11n, after logic fix (141 images)**

```
GT \ Pred    LOW   MEDIUM   HIGH
      LOW     23       14      7
   MEDIUM      0        0      2
     HIGH     23       15     57
```

**Table 8 — Confusion matrix: yolo26s, after logic fix (141 images)**

```
GT \ Pred    LOW   MEDIUM   HIGH
      LOW     17       20      7
   MEDIUM      0        0      2
     HIGH     15       21     59
```

#### Visualization Examples

> **Figure 3 — Example HIGH risk prediction (correct).**  
> `outputs/visualizations_yolo11n/HIGH/000000005862.jpg`  
> A dog detected in close proximity to a laptop. YOLO correctly localizes both objects; the depth gate confirms they are on the same depth plane; mischief score = 0.801 → HIGH. Warning: *"Mischief Alert! Dog is going for the laptop!"*

> **Figure 4 — Example LOW risk prediction (correct).**  
> `outputs/visualizations_yolo11n/LOW/000000003789.jpg`  
> Pet detected with no nearby household objects, or nearby objects separated by significant depth. Risk score = 0.0 → LOW. Status: *"All clear. Pet is behaving peacefully."*

> **Figure 5 — Failure case: HIGH predicted as LOW.**  
> See `outputs/visualizations_yolo11n/HIGH/` for examples where ground truth is HIGH but system outputs LOW.  
> Typical cause: monocular depth estimation produces near-equal depth values for all objects in a flat-depth scene (e.g., all objects on the same table), causing the depth gate to be uninformative. In these cases the system under-fires relative to the true risk.

> **Figure 6 — Failure case: LOW predicted as HIGH.**  
> See `outputs/visualizations_yolo11n/LOW/` for examples where ground truth is LOW but system outputs HIGH.  
> Typical cause: pet and object appear close in 2D but are on different furniture surfaces. The depth gate partially suppresses the score but not fully when depth map resolution is low relative to object size.

#### Analysis of Scoring Logic Impact

The accuracy improvement from 0.29→0.57 (after logic fix) arose primarily from **reducing false HIGH predictions**. Under the original additive scoring formula, depth contributed weakly as an additional term rather than as a gate. A cat anywhere in the frame with any detectable object would score HIGH if their 2D proximity was sufficient.

After converting depth to a multiplicative gate, predictions collapsed to near zero whenever the depth similarity between a pet and object dropped below a threshold — even if the 2D bounding boxes appeared close. This correctly suppresses the majority of false positives.

**Key finding:** Model size (yolo11n vs yolo26s) had negligible impact on risk accuracy (+/- 0.03), despite the 26s model having 2× more parameters. This strongly suggests the system's risk-classification bottleneck is the mischief scoring formula, not the object detection backbone.

---

## 6. Discussion

### 6.1 What Worked

**Depth as a multiplicative gate** was the single most impactful design decision. Converting depth from an additive term to a gate nearly doubled risk accuracy without any change to the detection backbone. This validates the core architectural insight that 3D reasoning must suppress 2D signals, not simply add to them.

**Two-phase fine-tuning (in progress)** addresses a concrete and well-understood failure mode — the learning rate conflict between a pretrained backbone and a randomly reinitialized detection head. The `exp_c_freeze` experiment is expected to produce meaningful mAP once the head stabilizes (typically within 5–10 epochs at lr=0.005).

### 6.2 Limitations

**Ordinal depth estimation.** Depth Anything V2 produces relative depth rankings, not metric distances. In scenes where all objects are at approximately the same physical depth (e.g., all on the same table surface), the depth gate provides no discriminating signal and the system reverts to pure 2D proximity — the exact limitation the depth pipeline was designed to overcome.

**Test set imbalance.** The 141-image test set contains 67.4% HIGH labels. A degenerate model predicting all images as HIGH would achieve accuracy 0.674 — higher than our best model at 0.567. Accuracy is therefore a misleading single-number summary. Macro-averaged F1 or balanced accuracy would be more informative for this skewed distribution.

**Fine-tuning incomplete.** The `exp_c_freeze` run was initiated during the final hours before submission. Whether fine-tuning improves mAP over the pretrained baseline (0.6489) remains to be determined. The initial exp_c failure (30 epochs, mAP=0) consumed significant time.

### 6.3 AI Assistance Disclosure

Generative AI (Claude, claude.ai/code) was used as a coding assistant throughout this project. Specific uses:

| Area | AI Assistance | My Verification and Correction |
|---|---|---|
| `scripts/03_train.py` extension | Generated `--fraction` and `--freeze` argument code | Verified correct mapping to Ultralytics `model.train()` params; tested end-to-end |
| MPS crash diagnosis | Initial hypothesis suggested `copy_paste` as cause | I disprovedthis by running `copy_paste=0` — crash persisted; I identified the true cause as the MPS `unique()` bug through stack trace analysis |
| MPS patch code | Suggested `.cpu()` move as fix | I added the `max(..., 0)` safety clamp and verified the fix across two crash locations |
| Per-class analysis structure | Suggested table format | I wrote all quantitative content, root-cause reasoning, and class-specific analysis based on my own interpretation of the metrics |
| `scripts/06_eval_risk_labels.py` | Assisted with script structure | I identified the class-index mismatch bug, specified the fix (name-based matching), and verified the corrected output against manual inspection |
| Report writing | Assisted with structure and phrasing | All technical claims, numerical results, and interpretations are my own; AI-generated text was revised for accuracy and specificity |

A specific limitation I observed: AI initially diagnosed the MPS crash as an augmentation parameter issue (`copy_paste`) without evidence. Running the controlled experiment (`copy_paste=0`, same crash) disproved this — demonstrating the importance of empirical validation over AI-suggested hypotheses. The correct root cause (MPS `unique()` returning garbage counts) required reading the PyTorch/Ultralytics source and understanding the MPS execution model, which I did myself.

---

## 7. Conclusion

The Pet Mischief Detector successfully classifies pet–object interaction risk using a two-model pipeline combining YOLO object detection and monocular depth estimation. My specific contributions spanned pipeline design, evaluation infrastructure, baseline experiments, per-class analysis, and fine-tuning.

The primary experimental finding is that the mischief scoring logic — specifically the treatment of depth as a multiplicative gate — has far greater impact on risk-classification accuracy than backbone size or fine-tuning. Accuracy improved from 0.29 to 0.57 through logic refinement alone.

The fine-tuning effort, while not yet yielding final results, produced important engineering contributions: an identified and patched Apple Silicon crash in the Ultralytics library, a diagnosis of a fundamental learning rate failure mode for reinitialized detection heads, and a corrected two-phase training design that is expected to improve weak-class detection (remote, cup) when completed.

---

## Appendix: Key Files

| File | Description |
|---|---|
| `imgs/two_objects_*.jpeg` | Reference images demonstrating 2D-vs-3D distortion |
| `scripts/03_train.py` | Fine-tuning script (I added `--fraction`, `--freeze`) |
| `scripts/04_evaluate.py` | Official YOLO mAP evaluation |
| `scripts/06_eval_risk_labels.py` | Risk-label accuracy evaluation (name-based matching) |
| `docs/finetuning_plan.md` | Per-class root-cause analysis and fine-tuning strategy |
| `docs/contribution.md` | Detailed individual contribution log |
| `outputs/eval/test_metrics.json` | Pretrained yolo26s mAP baseline |
| `outputs/eval/per_class_ap.png` | Per-class AP bar chart |
| `outputs/risk_eval_yolo11n.csv` | Risk-label results — yolo11n |
| `outputs/risk_eval_yolo26s.csv` | Risk-label results — yolo26s |
| `model/runs/exp_c/` | Exp C fine-tuning run (30 epochs, mAP=0 result) |
| `model/runs/exp_c_freeze/` | Exp C Freeze run (two-phase, ongoing) |
| `outputs/visualizations_yolo11n/` | Annotated output images — yolo11n |
| `outputs/visualizations_yolo26s/` | Annotated output images — yolo26s |
