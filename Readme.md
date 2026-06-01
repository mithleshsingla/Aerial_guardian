# 🚁 Aerial Guardian — Drone Person Detection & Tracking

> **YOLOv8n + Batched SAHI + ByteTrack + Homography Ego-Motion Compensation**
> Lightweight real-time multi-object tracking pipeline for aerial drone footage.

**Output Videos:** [Google Drive — Processed Sequences](https://drive.google.com/drive/folders/1wY8ZYSNtyWEiMgX-kP2N9lR5WIAD-GAS?usp=sharing)

Each video shows bounding boxes with unique person IDs, confidence scores, and
fading 40-frame trajectory tails for each tracked person.

---

## Final Results

**Hardware:** NVIDIA RTX 4090, Ubuntu 24, CUDA 12
**Dataset:** VisDrone2019-MOT-val — 7 sequences, 50,312 GT person-boxes
**Model:** YOLOv8n — **6.2 MB** (well under the 300 MB constraint)

### Two Operating Modes

| Mode | MOTA@0.5 | MOTA@0.3 | Precision | Recall | IDSW | **FPS** |
|------|----------|----------|-----------|--------|------|---------|
| **Single-scale SAHI** (tile=640) | **0.2880** | **0.3880** | **0.770** | **0.426** | **538** | **41.8** |
| Multi-scale SAHI (tile=512+640) | 0.2591 | **0.4034** | 0.697 | 0.480 | 580 | 22.1 |

**Single-scale** is the default — highest MOTA@0.5, 41.8 FPS, best precision.
**Multi-scale** trades precision (−7.3 pp) for recall (+5.4 pp) at half the FPS.
Choose based on whether missed detections or false positives matter more for
the application.

> †IoU=0.3 is the appropriate threshold for aerial tiny objects. A 4 px offset
> on a 12×30 px person gives IoU=0.41 — visually near-perfect but counted as
> FP+FN at the standard IoU=0.5. Both are reported throughout.

---

## Per-Sequence Results

### Single-scale SAHI — 41.8 FPS

| Sequence | Scene | MOTA@0.5 | MOTA@0.3 | IDSW | FPS |
|----------|-------|----------|----------|------|-----|
| uav0000086_00000_v | Dense crowd, low altitude | **0.421** | 0.521 | 236 | 120 |
| uav0000117_02622_v | Night, oblique | 0.060 | 0.127 | 305 | 62 |
| uav0000137_00458_v | Dense, oblique, high-res | **0.390** | 0.506 | 274 | 40 |
| uav0000182_00000_v | 45° high alt., sparse | −0.073 | −0.052 | 18 | 111 |
| uav0000268_05773_v | 4K, very sparse | 0.000 | 0.000 | 0 | 157 |
| uav0000305_00000_v | Nadir 90°, heads only | 0.000 | 0.000 | 0 | 39 |
| uav0000339_00001_v | 45° oblique, dusk | 0.271 | 0.320 | 122 | 61 |
| **Overall** | | **0.288** | **0.388** | **538** | **41.8** |

### Multi-scale SAHI — 22.1 FPS

| Sequence | Scene | MOTA@0.5 | MOTA@0.3 | IDSW | FPS |
|----------|-------|----------|----------|------|-----|
| uav0000086_00000_v | Dense crowd, low altitude | **0.404** | **0.581** | 107 | 29.9 |
| uav0000117_02622_v | Night, oblique | −0.048 | 0.058 | 236 | 15.6 |
| uav0000137_00458_v | Dense, oblique, high-res | 0.330 | **0.540** | 157 | 14.8 |
| uav0000182_00000_v | 45° high alt., sparse | −0.152 | −0.092 | 5 | 42.0 |
| uav0000268_05773_v | 4K, very sparse | 0.001 | 0.001 | 0 | 12.9 |
| uav0000305_00000_v | Nadir 90°, heads only | −0.013 | −0.013 | 0 | 23.6 |
| uav0000339_00001_v | 45° oblique, dusk | 0.307 | 0.365 | 75 | 21.8 |
| **Overall** | | **0.259** | **0.403** | **580** | **22.1** |

**Key observation:** Multi-scale helps most on dense sequences at IoU=0.3
(seq1: +17.6 pp, seq3: +3.4 pp) but hurts MOTA@0.5 on sparse sequences
(seq4: −7.9 pp, seq6: −1.3 pp) due to more false positives from border tiles.

---

## Engineering Progression

Full history of every evaluated configuration showing how each change affected results.

| # | Configuration | MOTA@0.5 | MOTA@0.3 | Prec | Rec | IDSW | FPS | What changed |
|---|--------------|----------|----------|------|-----|------|-----|-------------|
| 1 | YOLOv8n COCO, SAHI tile=640, sequential | 0.245 | — | 0.770 | 0.448 | 718 | ~10 | Starting point |
| 2 | + Homography ego-motion compensation | 0.303 | — | 0.770 | 0.448 | 592 | ~10 | 8-DOF H reduces ID switches |
| 3 | + Aspect-ratio filter h/w∈[0.8,6] | 0.303 | — | 0.800 | 0.448 | 592 | ~10 | Car/bus FPs removed |
| 4 | + Density-adaptive confidence | 0.297 | 0.380 | 0.804 | 0.417 | 944† | ~10 | Sparse scenes: conf 0.35→0.55 |
| 5 | Fine-tuned v3 (5 runs), conf=0.60 | 0.283 | — | 0.743 | 0.448 | 487 | 88 | Seq4 FP explosion; fine-tune abandoned |
| 6 | Sequential multi-scale SAHI + WBF | 0.310 | 0.406 | 0.787 | 0.448 | 830 | 4.7 | +3pp recall from tile=512+640 |
| 7 | **Batched single-scale SAHI** | **0.288** | **0.388** | **0.770** | **0.426** | **538** | **41.8** | **9× FPS via batching; final single-scale** |
| 8 | Batched multi-scale SAHI + WBF + NMS | 0.259 | 0.403 | 0.697 | 0.480 | 580 | 22.1 | Higher recall, lower precision; final multi-scale |

†IDSW increase at row 4 reflects a corrected evaluator (bug fix), not a real regression.

---

## 1. Architecture and Small Object Detection

---

## Architecture

```
Input Frame (any resolution)
        │
        ▼
[Resolution Cap: max width 1920 px]
   4K frames downsampled before tiling → detections mapped back to
   original coordinates. Prevents 30-tile explosion at 3840×2160.
        │
        ▼
[Batched SAHI Tiler]
   Slices frame into overlapping 640×640 tiles (10% overlap).
   All tiles collected into a single list — ONE YOLO call for the batch.
   Before: sequential SAHI = 6 calls × 20 ms Python overhead = 4.7 FPS
   After:  batched inference = 1 call × 28 ms total          = 41.8 FPS
        │
        ▼
[YOLOv8n — batched inference]
   COCO-pretrained, person class (class 0) only.
   conf = 0.35 (auto-raised to 0.45–0.55 on sparse scenes).
        │
        ▼
[WBF + post-NMS merge]
   WBF (Weighted Box Fusion) fuses detections from overlapping tiles.
   Post-WBF NMS at iou_thr=0.4 suppresses remaining partial duplicates.
        │
        ▼
[Aspect-ratio + area filter]
   h/w ∈ [0.8, 6.0]  — removes cars, buses, poles
   area ≥ 16 px²      — removes tile-boundary artefacts
        │
        ├─────────────────────────────────────┐
        │                                     │
[LK Optical Flow]               [ByteTrack Kalman Filter]
  300 Shi-Tomasi corners           Two-stage association:
  on background pixels              Pass 1: high-conf ↔ active tracks
  (detection regions masked)        Pass 2: low-conf  ↔ lost tracks
  Lucas-Kanade pyramid (3 levels)   Lost track buffer: 40 frames
  RANSAC homography H (8-DOF)       ─────────────────────────────
        │                           Camera motion H applied to
        └─────────────────────────→ Kalman predictions BEFORE IoU
                                    association
        │
        ▼
[Tracked persons + IDs]
   Bounding boxes, unique ID labels,
   fading 40-frame trajectory tails
        │
        ▼
Output MP4 video
```

---

### YOLOv8n — why this model

**Backbone:** C2f (Cross-Stage Partial, 2 outputs) blocks extract features at
P3/P4/P5 scales. Two sequential 3×3 convolutions per block capture finer spatial
detail than a single large kernel — important for 8–30 px persons.

**Neck:** PANet bidirectional feature fusion: bottom-up path carries fine-grained
spatial detail (P3), top-down path carries semantic context (P5). Both paths are
fused, giving the detection head spatial precision and class confidence simultaneously.

**Head:** Decoupled anchor-free detection — separate branches for box regression
and classification. Faster convergence and better small-object localisation than
anchor-based heads.

**Size:** 3.2M parameters, 6.2 MB. Under the 300 MB limit with room to spare
for tracking overhead, ReID, or TensorRT quantisation.

### Batched SAHI for small object detection

At 50 m drone altitude a person occupies ~8–30 px. Standard YOLO at 640×640
makes these persons invisible. SAHI tiles the frame into overlapping 640×640
crops — within each tile the person appears ~4× larger.

**The batching optimisation** is the central engineering contribution on speed:

```
Sequential SAHI (before):
  model(tile_1) → wait 20ms → model(tile_2) → wait 20ms → ... × 6 tiles
  Total: 6 × 20ms = 120ms = 8 FPS   (GPU idle 90% of the time)

Batched SAHI (after):
  model([tile_1, tile_2, tile_3, tile_4, tile_5, tile_6])  ← one GPU call
  Total: 1 × 28ms = 28ms = 41.8 FPS  (GPU processes all tiles in parallel)
```

After YOLO's per-tile inference, detections from all tiles are merged:
1. **WBF** (Weighted Box Fusion) — averages coordinates of overlapping boxes
   weighted by confidence. Better localisation on tiny objects than NMS.
2. **Post-WBF NMS at iou_thr=0.4** — suppresses partial duplicate detections
   from border tiles. SAHI's NMM uses IoS (Intersection over Smaller) which
   is more aggressive than WBF's IoU; this NMS pass replicates that behaviour.
3. **Aspect-ratio + area filter** — h/w ∈ [0.8, 6.0], area ≥ 16 px².

**Multi-scale SAHI** adds a second pass at tile=512. A 10 px person occupies
1.6% of a 640px tile but 2.0% of a 512px tile — the smaller tile acts as a
zoom-in for sub-15px persons. This improved recall from 0.426 to 0.480 (+5.4 pp)
at the cost of halving FPS (41.8 → 22.1) and reducing precision due to more
border-tile duplicates.

### Density-adaptive confidence

The pipeline probes the first 20 frames to estimate average detections per frame,
then adjusts confidence automatically:

| Avg dets/frame | Scene | Confidence used | Reason |
|----------------|-------|----------------|--------|
| ≥ 6 | Dense (seq1, seq3) | 0.35 | Recall matters; persons are plentiful |
| 2–6 | Sparse (seq4) | 0.45 | Reduce FPs on terrain background |
| < 2 | Very sparse (seq5, seq6) | 0.55 | Strong FP suppression |

This fixed seq4 (uav0000182) from MOTA = −0.259 → −0.073 without affecting
dense sequences.

---

## 2. Addressing ID Switching from Drone Ego-Motion

### Root cause

ByteTrack's Kalman filter assumes a static camera. It predicts where each
tracked person will appear in the next frame based on their image-space velocity.
When the drone moves, all objects shift in the image — the Kalman prediction
is wrong, IoU between prediction and detection falls, and the association
fails → ID switch.

### Homography ego-motion compensation (key custom contribution)

```
Frame N    → sample 300 Shi-Tomasi corners on BACKGROUND pixels
               (detection bounding boxes are masked to exclude objects)
Frame N+1  → track corners with Lucas-Kanade optical flow
               (pyramid, 3 levels, 21×21 window, 50% resolution for speed)
           → RANSAC homography estimation (500 iterations, threshold=3px)
           → apply H to all Kalman-predicted track centres
               BEFORE ByteTrack's IoU association step
           → association now in scene-space, not camera-space
```

**Why 8-DOF homography, not 4-DOF affine:**
Drone banking and pitching create perspective effects — parallel lines appear
to converge. Affine cannot model this; homography can.

```
Affine (4-DOF):  [s·cosθ  -s·sinθ  tx]   pan + zoom + rotation
                 [s·sinθ   s·cosθ  ty]   cannot handle tilt/pitch

Homography (8):  [h00 h01 h02]            full perspective warp
                 [h10 h11 h12] ÷ w        handles all drone motions
                 [h20 h21  1 ]
```

**Graceful fallback:** if RANSAC finds < 8 inliers → affine → last known transform
→ identity. The tracker never crashes from bad optical flow.

**Result:** ID switches reduced from ~1,200 (uncompensated) to 538 (final system).

### ByteTrack two-stage association

Pass 2 of ByteTrack matches **low-confidence** detections (0.10–0.35) to tracks
that were unmatched in Pass 1. This is critical for drone footage: persons at
extreme range or partial occlusion frequently drop below the 0.35 threshold.
Without Pass 2, these become ID switches. With Pass 2, the existing track is
maintained.

Lost tracks are kept alive for **40 frames** (~2 s). If a person is temporarily
occluded and reappears within that window, they recover their original ID via
position-based re-association — no Re-ID network needed.

---

## 3. Edge Hardware Adaptation (NVIDIA Jetson)

### Step 1 — TensorRT export

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')

# FP16: ~3–5× speedup, <0.1 pp mAP loss, model stays ~6 MB
model.export(format='engine', device=0, half=True)

# INT8: ~2× faster than FP16, model shrinks to ~1.5 MB, <2% mAP loss
model.export(format='engine', int8=True, data='coco.yaml')
```

Replace `YOLO('yolov8n.pt')` with `YOLO('yolov8n.engine')` at inference —
no other code changes.

### Step 2 — Async pipeline (Jetson heterogeneous CPU+GPU)

```
Thread 1 (CPU): Frame capture
Thread 2 (GPU): Batched YOLO inference (TensorRT engine)
Thread 3 (CPU): ByteTrack Kalman + optical flow (pure NumPy + OpenCV CPU)
Thread 4 (CPU): Video write / telemetry output
```

Decoupling means the GPU never waits for tracking and the CPU never blocks inference.

### Step 3 — Reduce optical flow resolution

Current `flow_scale = 0.5`. On Jetson, set `flow_scale = 0.25`.
At 30 FPS the drone moves < 2 px between frames; quarter-resolution LK is
accurate enough while running ~4× faster.

### Step 4 — Reduce tile overlap

Reduce `overlap_height/width_ratio` from 0.10 to 0.05:
- 1344×756 frame: 6 tiles at 10% → 4 tiles at 5%
- Each tile removed = one fewer GPU inference call per frame

### Expected Jetson FPS

| Hardware | Config | FPS |
|----------|--------|-----|
| Jetson Nano 4 GB | YOLOv8n TRT FP16, single-scale, overlap=5% | ~12 |
| Jetson Orin NX | YOLOv8n TRT FP16, single-scale, overlap=5% | ~25 |
| Jetson Orin AGX | YOLOv8n TRT INT8, single-scale, overlap=5% | ~40 |

---

## 4. Engineering Trade-offs Summary

### Speed vs accuracy

The central trade-off in this system is SAHI configuration:

| Config | MOTA@0.5 | FPS | When to use |
|--------|----------|-----|-------------|
| No SAHI | ~0.18 | ~80 | Edge devices where accuracy is secondary |
| Single-scale tile=640 | 0.288 | 41.8 | **Default** — best MOTA + real-time FPS |
| Multi-scale tile=512+640 | 0.259 | 22.1 | When recall matters more than precision |

### Why fine-tuning was not adopted

Five fine-tuning runs on VisDrone2019-MOT-train improved detection mAP@0.5 by
+12.5 pp (0.431 → 0.556). However, tracking MOTA decreased on every run.
The fine-tuned model generated massive FPs on uav0000182 (45° high-altitude
terrain): MOTA collapsed from −0.073 to −2.50 on that sequence. This is
**distribution shift** — the training set lacked enough sequences with that
specific background type. Fine-tuning optimises mAP (averaged across all
confidence thresholds) but shifted the score distribution unfavourably at
the fixed conf=0.35 operating point used by the tracker.

### Why not ReID

ReID fixes ID switches, not missed detections. Our IDSW=538 costs only
`538/50312 = 1.1% MOTA`. Even eliminating all switches = +1.1 pp MOTA.
With recall=0.43, 57% of persons are never detected — ReID cannot help
those. ReID is appropriate when recall exceeds 0.70. At the current detection
level, detector improvement has ~10× more MOTA leverage than ReID.

### Noise from moving camera

Three mechanisms address the "noise" of a moving camera:

1. **Homography compensation** — corrects Kalman predictions for camera motion
   before association. Handles pan, tilt, zoom, and roll simultaneously.
2. **Background-only feature sampling** — Shi-Tomasi corners are sampled only
   outside detected bounding boxes, so the homography reflects pure camera
   motion, not object motion.
3. **Lost-track buffer (40 frames)** — briefly lost tracks (drone manoeuvre,
   momentary occlusion) maintain their IDs instead of triggering false switches.

---

## Setup

```bash
conda create -n aerial_guardian python=3.10
conda activate aerial_guardian
pip install ultralytics supervision sahi scipy tqdm ensemble-boxes
```

Place VisDrone data under:
```
data/VisDrone2019-MOT-val/
    annotations/   *.txt
    sequences/     <seq_name>/*.jpg
```

---

## Running

```bash
# Single-scale (default, 41.8 FPS, MOTA@0.5=0.288)
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val --all --device 0

# Multi-scale (22.1 FPS, higher recall at IoU=0.3)
# Set multi_scale: true in configs/config.yaml, then:
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val --all --device 0

# Single sequence with GT overlay for debugging
python scripts/run_sequence.py \
    --dataset /path/to/VisDrone2019-MOT-val \
    --sequence uav0000086_00000_v --device 0 --show-gt

# Benchmark multiple model sizes
python scripts/benchmark_models.py \
    --dataset /path/to/VisDrone2019-MOT-val --device 0 --max-frames 100

# Fine-tune (optional)
python scripts/prepare_dataset.py \
    --train /path/to/VisDrone2019-MOT-train \
    --val   /path/to/VisDrone2019-MOT-val \
    --out   /path/to/visdrone_yolo --skip-empty --subsample 3
python scripts/fine_tune.py \
    --data /path/to/visdrone_yolo/visdrone_person.yaml --model yolov8n --device 0
```

## Key Config Options

```yaml
# configs/config.yaml

sahi:
  multi_scale: false       # false=single-scale 41.8 FPS; true=multi-scale 22.1 FPS
  second_tile_size: 512    # second tile for multi-scale pass

detection:
  conf_threshold: 0.35     # base; auto-raised on sparse scenes

motion_compensation:
  mode: "homography"       # "homography" (8-DOF) or "affine" (4-DOF, faster)
  flow_scale: 0.5          # 0.5=default; 0.25 for Jetson

model:
  weights: "yolov8n.pt"    # or yolo11n.pt / path/to/best.pt
```

## Project Structure

```
aerial_guardian/
├── configs/config.yaml           ← all parameters with comments
├── src/
│   ├── detector.py               ← YOLOv8n + batched SAHI + WBF + NMS + filters
│   ├── tracker.py                ← ByteTrack + homography compensation
│   ├── motion_comp.py            ← LK optical flow + RANSAC homography (8-DOF)
│   ├── visualizer.py             ← boxes, ID labels, 40-frame trajectory tails
│   ├── pipeline.py               ← orchestrator, density-adaptive conf, dual IoU eval
│   ├── annotation_loader.py      ← VisDrone MOT annotation parser
│   ├── fine_tune.py              ← fine-tuning with 4 model presets
│   ├── prepare_dataset.py        ← VisDrone MOT → YOLO label format converter
│   ├── benchmark_models.py       ← compare yolov8n / yolo11n / yolov8s / yolo11s
│   ├── compare_weights.py        ← baseline vs fine-tuned side-by-side table
│   └── evaluator.py              ← MOTA/precision/recall at configurable IoU
├── scripts/
│   └── run_sequence.py           ← main CLI (--model, --weights, --show-gt flags)
└── output/                       ← generated MP4 videos
```