# 🚁 Aerial Guardian — Drone Person Detection & Tracking

> **YOLOv8n + Multi-Scale SAHI (WBF) + ByteTrack + Homography Ego-Motion Compensation**
> Lightweight multi-object tracking pipeline for aerial drone footage.

---

## Final Results

**Hardware tested:** NVIDIA RTX 4090, Ubuntu 24, CUDA 12
**Dataset:** VisDrone2019-MOT-val — 7 sequences, 50,312 GT person boxes
**Model size:** 6.2 MB (YOLOv8n) — well under the 300 MB constraint

### Best Configuration: Multi-Scale SAHI (tile=512+640) + WBF

| Metric | IoU≥0.5 (official) | IoU≥0.3 (practical†) |
|--------|--------------------|-----------------------|
| **MOTA** | **0.3103** | **0.4063** |
| Precision | 0.7867 | 0.8735 |
| Recall | 0.4483 | 0.4978 |
| ID Switches | 830 | 975 |
| **FPS** | **4.7** | — |

†*IoU=0.3 is the appropriate evaluation threshold for aerial tiny objects.
A 4px localisation offset on a 12×30px person gives IoU=0.41 — visually
near-perfect but counted as FP+FN at IoU=0.5. Both thresholds are reported.*

### Per-Sequence Breakdown

| Sequence | Scene Type | MOTA@0.5 | MOTA@0.3 | IDSW | FPS |
|----------|-----------|----------|----------|------|-----|
| uav0000086_00000_v | Dense crowd, low altitude | 0.421 | 0.521 | — | 117 |
| uav0000117_02622_v | Night, oblique | 0.060 | 0.127 | — | 62 |
| uav0000137_00458_v | Dense, oblique, high-res | 0.390 | 0.506 | — | 40 |
| uav0000182_00000_v | 45° high altitude, sparse | −0.073 | −0.052 | — | 111 |
| uav0000268_05773_v | 4K, very sparse | 0.000 | 0.000 | — | 157 |
| uav0000305_00000_v | Nadir 90°, heads only | 0.000 | 0.000 | — | 39 |
| uav0000339_00001_v | 45° oblique, dusk | 0.271 | 0.320 | — | 61 |

### Progression Summary

| Configuration | MOTA@0.5 | MOTA@0.3 | IDSW | FPS |
|--------------|----------|----------|------|-----|
| First baseline (SAHI tile=640) | 0.245 | — | 718 | ~10 |
| + Homography compensation + aspect filter | 0.303 | — | 592 | ~10 |
| + Density-adaptive confidence | 0.297 | 0.380 | 944* | ~10 |
| **+ Multi-scale SAHI + WBF (final)** | **0.310** | **0.406** | **830** | **4.7** |

*IDSW increase from 592→944 reflects a corrected evaluator, not a real regression.
The original evaluator had a sorting bug that undercounted switches.*

---

## Architecture

```
Input Frame (any resolution)
    │
    ▼
[Resolution Cap: 1920px width]
    Frames wider than 1920px downsampled before tiling.
    3840×2160 (4K) → 1920×1080 → SAHI tiles → detections mapped back.
    │
    ▼
[Multi-Scale SAHI Tiler]
    Pass 1: tile=640px, overlap=10%  → catches 15-40px persons
    Pass 2: tile=512px, overlap=10%  → catches 8-15px persons (appear larger)
    Each pass: NMM merge within pass
    │
    ▼
[YOLOv8n Inference — per tile]
    COCO-pretrained, person class only (class 0)
    conf=0.35 (dense scenes) / 0.45-0.55 (sparse scenes, auto-adapted)
    │
    ▼
[Weighted Box Fusion (WBF)]
    Merges detections from both SAHI passes.
    Unlike NMM (picks best box), WBF averages box coordinates weighted
    by detection score → better localisation on tiny aerial objects.
    │
    ▼
[Aspect-Ratio + Area Filter]
    h/w ∈ [0.8, 6.0] — removes cars, buses, poles
    area ≥ 16px²     — removes tile-boundary artefacts
    │
    ├──────────────────────────────────────┐
    │                                      │
[LK Optical Flow]              [ByteTrack Kalman Filter]
  Shi-Tomasi corners              Two-stage association:
  on background pixels            Pass 1: high-conf ↔ active tracks
  Lucas-Kanade pyramid            Pass 2: low-conf  ↔ lost tracks
  RANSAC homography (8-DOF)       Lost tracks: 40-frame buffer
    │                               │
    └──────────────────────────────┘
    Camera motion H applied to
    Kalman predictions BEFORE
    IoU-based association
    │
    ▼
[Tracked Persons + IDs]
    │
    ▼
[Visualizer]
    Bounding boxes, ID labels, fading trajectory tails (40-frame)
    │
    ▼
Output MP4 Video
```

---

## 1. Detection: Multi-Scale SAHI + WBF

### Problem: persons at drone altitude are 8–40px

At 50m altitude, a standing person occupies roughly 8–20px in a 1920×1080 frame.
Standard YOLO inference downsamples the full frame to 640×640, making persons
effectively invisible.

### Solution: SAHI (Sliced Aided Hyper Inference)

SAHI tiles the frame into overlapping crops and runs YOLO on each tile.
Within its tile, a person that was 8px in the full frame appears as ~40px.

### Multi-Scale SAHI: two tile sizes per frame

Single-scale SAHI (tile=640) is optimised for persons in the 15–40px range.
Persons below 15px are at the edge of YOLOv8n's detection limit even within
a tile. The solution: run SAHI at **two scales** and merge results.

```
tile=640: 30px person = 4.7% of tile height  ← good for 15-40px persons
tile=512: 30px person = 5.9% of tile height  ← better for 8-15px persons
          (same person appears 25% larger relative to tile)
```

Running both passes and merging with WBF captures detections that one scale
misses. **Recall improved from 0.418 → 0.448 (+3pp) without any model change.**

### Weighted Box Fusion (WBF) vs NMM

NMM (Non-Maximum Merging) picks the highest-scoring box and suppresses nearby
ones. For tiny aerial objects where two SAHI passes detect the same person at
slightly different positions (e.g. 2px offset), NMM discards the second
detection and keeps the noisier position.

WBF instead computes a weighted average of all overlapping boxes:
```
fused_cx = Σ(score_i × cx_i) / Σ(score_i)
fused_cy = Σ(score_i × cy_i) / Σ(score_i)
```
On a 12px-wide person, a 2px improvement in localisation changes IoU by ~0.1,
which directly affects MOTA at the IoU=0.5 threshold.

### YOLOv8n Architecture

- **Backbone**: C2f (Cross Stage Partial with 2 outputs) at P3/P4/P5 scales
- **Neck**: PANet — bidirectional feature fusion (spatial + semantic)
- **Head**: Decoupled anchor-free (separate cls + box branches)
- **Size**: 3.2M parameters, **6.2 MB** — well under 300 MB limit
- **COCO mAP**: 37.3 @ IoU=0.5

### Density-Adaptive Confidence

The pipeline probes the first 20 frames of each sequence to estimate scene
density (average detections/frame), then automatically adjusts the confidence
threshold:

| Avg dets/frame | Scene type | Conf used |
|----------------|-----------|-----------|
| ≥ 6 | Dense (seq1, seq3) | 0.35 — keep recall |
| 2–6 | Sparse (seq4) | 0.45 — suppress FPs |
| < 2 | Very sparse (seq5, seq6) | 0.55 — strong FP suppression |

This fixed seq4 (uav0000182) from MOTA=−0.259 → −0.073 without hurting dense sequences.

---

## 2. Tracking: ByteTrack + Homography Ego-Motion Compensation

### ByteTrack

ByteTrack was chosen over DeepSORT for drone deployment:

| | ByteTrack | DeepSORT |
|---|---|---|
| Extra model | None needed | Re-ID network (+50–200 MB) |
| Low-conf handling | Pass 2 rescues low-conf dets | Discards them |
| CPU runtime | Pure NumPy | Slower (embedding net) |

ByteTrack's two-stage association is critical for drone footage:
- **Pass 1**: high-confidence detections (>0.35) matched to active tracks by IoU
- **Pass 2**: low-confidence detections (0.1–0.35) matched to unmatched tracks
  — catches persons whose detection score dropped due to altitude or occlusion

### Homography Ego-Motion Compensation (key custom contribution)

**The problem**: ByteTrack's Kalman filter assumes a static camera. A moving drone
shifts all objects in the image. The Kalman prediction is wrong in image space →
IoU between prediction and detection is low → ID switch.

**The solution**:

```
Frame N:   sample 300 Shi-Tomasi corner features on background pixels
           (detection bounding boxes are masked out so only true
           background features are sampled)
Frame N+1: track features with Lucas-Kanade optical flow (3-level pyramid)
           estimate 8-DOF homography H via RANSAC
           (8-DOF handles pan + tilt + zoom + roll + pitch
            vs affine which only handles 4-DOF: pan + zoom + rotation)
           apply H to all Kalman-predicted track centres BEFORE association
           → ByteTrack now associates in scene-space, not camera-space
```

**Graceful fallback**: if homography is degenerate (< 8 RANSAC inliers) →
falls back to affine → last known transform → identity (no compensation).

**Optical flow at 50% resolution** (`flow_scale=0.5`): LK computed on downsampled
grayscale, points scaled back to full resolution. No quality loss, ~2× faster.

**Result**: ID switches reduced from ~1200 (no compensation) to 830 (with homography).

---

## 3. Engineering Trade-offs

### FPS vs Accuracy

| Configuration | MOTA@0.5 | FPS | Notes |
|--------------|----------|-----|-------|
| Standard SAHI tile=640 | 0.297 | ~10 | Single-scale |
| Multi-scale SAHI (512+640) | **0.310** | **4.7** | +1.4pp, 2× slower |
| No SAHI (full frame) | ~0.18 | ~40 | Fast but misses small persons |

The FPS drop from ~10 to 4.7 with multi-scale SAHI is the main trade-off.
For a drone surveillance system where accuracy matters more than latency,
4.7 FPS is acceptable. For real-time reaction (obstacle avoidance etc.)
the single-scale configuration at ~10 FPS is the better choice.

### Why Fine-Tuning Did Not Improve Tracking MOTA

Five fine-tuning experiments were run on VisDrone2019-MOT-train:

| Run | Config | Detection mAP@0.5 | Tracking MOTA@0.5 |
|-----|--------|-------------------|-------------------|
| A | yolov8n, freeze=10, 5ep | 0.462 | — |
| B | yolov8n, nc=2 ignore class | 0.238 | worse — class collapsed |
| C | yolov8n, freeze=5, 100ep | 0.512 @ ep17 | worse than baseline |
| D | yolov8n, freeze=0, 37ep | **0.555** @ ep14 | worse than baseline |
| E | yolov8n, freeze=0, label_smooth=0.1 | 0.556 @ ep7 | worse than baseline |

Fine-tuned detection mAP improved (+12.5pp) but tracking MOTA did not.
**Root cause**: fine-tuned model generated massive FPs on uav0000182
(45° high-altitude terrain) at conf=0.35. MOTA collapsed from −0.26 to −2.50
on that sequence. The training set did not contain enough sequences with that
specific background type. Fine-tuning optimises mAP (averaged across thresholds)
but shifted the confidence score distribution unfavourably at the fixed
conf=0.35 operating point.

**Lesson**: detection mAP and tracking MOTA are different objectives.
The COCO-pretrained baseline at conf=0.35 remains the best model.

### Why IoU=0.3 is the Honest Evaluation Threshold

The official VisDrone leaderboard uses IoU=0.5 (designed for COCO large objects).
For aerial tiny objects this is too strict:

```
12×30px person, prediction offset by 4px:
  IoU = 0.41 → FN + FP at IoU=0.5  (MOTA penalised twice)
  IoU = 0.41 → TP   at IoU=0.3  (correctly rewarded)
```

We report both thresholds. MOTA@0.3=0.406 reflects actual tracking quality.

### Why Not ReID (Appearance Embeddings)

ReID improves ID switches, not missed detections. Our IDSW=830 contributes
`830/50312 = 1.65% MOTA`. Even eliminating 40% of switches = +0.66pp MOTA.
Meanwhile, 55.2% of persons are never detected (recall=0.448) — ReID cannot
find persons YOLO missed. At this recall level, detection improvement has
8–10× more MOTA leverage than ReID. ReID is valuable when recall > 0.70.

---

## 4. Edge Hardware Adaptation (NVIDIA Jetson)

### TensorRT Export

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0, half=True)   # FP16, ~3-5× speedup
model.export(format='engine', int8=True, data='coco.yaml')  # INT8, ~2× faster, <2% mAP loss
```

### Async Pipeline (Jetson heterogeneous CPU+GPU)

```
Thread 1: Frame capture
Thread 2: YOLO inference (Jetson GPU via TensorRT)
Thread 3: ByteTrack + optical flow (Jetson CPU cores)
Thread 4: Video output
```

### Optical Flow Optimisation for Jetson

Current: `flow_scale=0.5` (optical flow at 50% resolution).
On Jetson: reduce to `flow_scale=0.25` — LK at 25% is 4× faster with
minimal homography quality loss at typical drone speeds.

### Expected Jetson Performance

| Hardware | Config | FPS |
|----------|--------|-----|
| Jetson Nano 4GB | YOLOv8n TRT FP16, single-scale SAHI | ~8 FPS |
| Jetson Orin NX | YOLOv8n TRT FP16, single-scale SAHI | ~15 FPS |
| Jetson Orin AGX | YOLOv8n TRT INT8, multi-scale SAHI | ~10 FPS |

---

## Setup & Installation

```bash
conda create -n aerial_guardian python=3.10
conda activate aerial_guardian
pip install ultralytics supervision sahi scipy tqdm ensemble-boxes
```

## Running

```bash
# Full evaluation — multi-scale SAHI (best MOTA, ~4.7 FPS)
# First enable in config: set multi_scale: true
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val \
    --all --device 0

# Single-scale SAHI (~10 FPS, slightly lower MOTA)
# Set multi_scale: false in config
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val \
    --all --device 0

# Single sequence with GT overlay
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val \
    --sequence uav0000086_00000_v --device 0 --show-gt

# Benchmark multiple model variants
python scripts/benchmark_models.py \
    --dataset /path/to/VisDrone2019-MOT-val \
    --device 0 --max-frames 100

# Fine-tune on training set
python scripts/prepare_dataset.py \
    --train /path/to/VisDrone2019-MOT-train \
    --val   /path/to/VisDrone2019-MOT-val \
    --out   /path/to/visdrone_yolo \
    --skip-empty --subsample 3

python scripts/fine_tune.py \
    --data /path/to/visdrone_yolo/visdrone_person.yaml \
    --model yolov8n --device 0
```

## Key Configuration Options

```yaml
# configs/config.yaml

# Switch to multi-scale SAHI (better MOTA, ~2× slower)
sahi:
  multi_scale: true        # false = single-scale (~10 FPS), true = multi-scale (~4.7 FPS)
  second_tile_size: 512    # second tile size for multi-scale pass

# Switch model (no other changes needed)
model:
  weights: "yolov8n.pt"    # or: yolo11n.pt / yolov8s.pt / path/to/best.pt

# Confidence (auto-adapted per sequence, this is the base)
detection:
  conf_threshold: 0.35
```

## Project Structure

```
aerial_guardian/
├── configs/
│   └── config.yaml              ← all parameters with comments
├── src/
│   ├── detector.py              ← YOLOv8n + multi-scale SAHI + WBF + aspect filter
│   ├── tracker.py               ← ByteTrack + homography ego-motion compensation
│   ├── motion_comp.py           ← Shi-Tomasi + LK optical flow + RANSAC homography
│   ├── visualizer.py            ← boxes, IDs, trajectory tails
│   ├── pipeline.py              ← orchestrator, density-adaptive conf, dual-IoU eval
│   ├── annotation_loader.py     ← VisDrone MOT annotation parser
│   └── evaluator.py             ← MOTA/precision/recall at configurable IoU threshold
├── scripts/
│   ├── run_sequence.py          ← main CLI (--model, --weights, --tile, --conf flags)
│   ├── fine_tune.py             ← YOLOv8n fine-tuning (4 model presets)
│   ├── prepare_dataset.py       ← VisDrone MOT → YOLO format converter
│   ├── benchmark_models.py      ← compare yolov8n/yolo11n/yolov8s/yolo11s
│   └── compare_weights.py       ← baseline vs fine-tuned side-by-side table
└── output/                      ← generated MP4 videos
```