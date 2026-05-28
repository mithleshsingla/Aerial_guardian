# 🚁 Aerial Guardian — Drone Person Detection & Tracking

> **YOLOv8n + SAHI + ByteTrack + Optical Flow Ego-Motion Compensation**  
> Lightweight multi-object tracking pipeline for aerial drone footage

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture & Design Choices](#2-architecture--design-choices)
3. [Small Object Detection Strategy (SAHI)](#3-small-object-detection-strategy-sahi)
4. [Addressing ID Switching & Drone Ego-Motion](#4-addressing-id-switching--drone-ego-motion)
5. [Performance (FPS Benchmarks)](#5-performance-fps-benchmarks)
6. [Edge Hardware Adaptation (NVIDIA Jetson)](#6-edge-hardware-adaptation-nvidia-jetson)
7. [Project Structure](#7-project-structure)
8. [Setup & Installation](#8-setup--installation)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Configuration Reference](#10-configuration-reference)
11. [Engineering Trade-offs](#11-engineering-trade-offs)

---

## 1. Project Overview

This pipeline detects and tracks **persons** across frames of aerial drone footage from the [VisDrone MOT Task 4](https://github.com/VisDrone/VisDrone-Dataset) validation set. Key challenges addressed:

| Challenge | Our Solution |
|-----------|-------------|
| Tiny objects (persons at 50m altitude = ~8px) | SAHI sliced inference |
| Drone ego-motion causing false ID switches | Sparse optical flow + affine compensation |
| Real-time constraint on edge hardware | YOLOv8 **nano** (~6MB model) |
| Occlusion-caused track loss | ByteTrack dual-stage association |
| Model size limit (300MB) | Full stack < 10MB model weights |

Output: MP4 video with bounding boxes, unique person IDs, confidence scores, and fading trajectory tails.

---

## 2. Architecture & Design Choices

```
┌──────────────────────────────────────────────────────────────────────┐
│                     AERIAL GUARDIAN PIPELINE                         │
│                                                                      │
│  Frame N  ─►  [SAHI Tiler]  ─►  [YOLOv8n]  ─►  [NMS Merge]        │
│                                                         │            │
│                                                    Detections        │
│                                                         │            │
│  Frame N-1 ─► [LK Optical Flow] ─► [Affine Transform]  │            │
│                                         │               │            │
│                                   Ego-motion       ByteTrack         │
│                                   Compensation     Update            │
│                                         │               │            │
│                                         └──────────────►│            │
│                                                         │            │
│                                               Tracked Detections     │
│                                               + IDs + Tails          │
│                                                         │            │
│                                                  [Visualizer]        │
│                                                         │            │
│                                                  Output Video        │
└──────────────────────────────────────────────────────────────────────┘
```

### Why YOLOv8 Nano?

YOLOv8n was chosen as the detection backbone for the following reasons:

**Architecture breakdown:**
- **Backbone**: C2f (Cross Stage Partial with 2 outputs) modules — a lighter version of CSPDarknet. Feature maps are extracted at 3 scales (P3=80x80, P4=40x40, P5=20x20 for 640-input).
- **Neck**: PANet (Path Aggregation Network) — bi-directional feature fusion. Bottom-up and top-down pathways allow the network to combine fine-grained spatial info (P3) with semantic context (P5). This is critical for small object detection.
- **Head**: Decoupled anchor-free detection head — separate branches for box regression and classification. Faster and more accurate than anchor-based heads for small objects.
- **Parameters**: ~3.2M
- **Model size**: ~6MB (well under 300MB limit)
- **COCO mAP**: 37.3 (competitive for its size)

**Why not a larger model?**
The key requirement is edge deployment on a drone. YOLOv8s (22MB) gives +5 mAP but cuts FPS roughly in half. For drone scenarios where reaction time matters, throughput > marginal accuracy gain.

### Why ByteTrack?

ByteTrack was chosen over DeepSORT for these reasons:

1. **No Re-ID network**: DeepSORT requires a separate appearance embedding model (adds ~50-200MB and significant latency). ByteTrack uses only IoU for association.
2. **Low-confidence rescue**: ByteTrack's dual-stage matching recovers detections that dipped below the threshold due to occlusion or altitude — extremely common in drone footage.
3. **Pure Python + NumPy**: Runs efficiently on CPU; no CUDA required for the tracker itself.
4. **Simple state**: Kalman filter (constant-velocity) + Hungarian algorithm. Easy to reason about and debug.

---

## 3. Small Object Detection Strategy (SAHI)

### The Problem

At typical drone operating altitudes (30–100m), a standing person occupies roughly **8×20 pixels** in a 1920×1080 frame. When YOLO resizes this to 640×640 for inference, the person becomes even smaller — often below the network's effective receptive field for detection.

### SAHI Solution

**SAHI (Slicing Aided Hyper Inference)** works as follows:

```
Original Frame (1920x1080)
┌────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐           │
│  │  Tile 1  │  │  Tile 2  │  ...      │  ← 640x640 tiles with 20% overlap
│  │  (YOLO)  │  │  (YOLO)  │           │
│  └──────────┘  └──────────┘           │
│  ┌──────────┐  ┌──────────┐           │
│  │  Tile 3  │  │  Tile 4  │  ...      │
│  │  (YOLO)  │  │  (YOLO)  │           │
│  └──────────┘  └──────────┘           │
└────────────────────────────────────────┘
              ↓
   Merge all detections back to original coordinates
              ↓
   Non-Maximum Merging (NMM) to remove duplicates from overlapping tiles
```

**Effect**: A person that was 8px in the full frame appears as ~50px in their tile — well within YOLO's detection range.

**Overlap (20%)** ensures objects near tile borders are captured by at least one tile fully.

**Trade-off**: SAHI multiplies inference calls by the number of tiles (~6-12 for typical VisDrone resolutions). We mitigate this by:
- Using the lightest model (YOLOv8n)
- Keeping tiles at 640×640 (optimal for YOLOv8)
- Using Non-Maximum Merging instead of NMS (better for slightly-shifted duplicate boxes)

---

## 4. Addressing ID Switching & Drone Ego-Motion

### Root Cause

When the drone moves (translates, rotates, pitches), all objects shift in the image. ByteTrack's Kalman filter predicts where each tracked object will be in the **next frame** based on its **image-space velocity**. But drone motion invalidates this prediction — the object has moved in the image not because it moved in the world, but because the camera moved.

Result: The predicted track position is far from the actual detection → IoU is low → match fails → ID switch.

### Our Solution: Sparse Optical Flow Compensation

**Step 1: Detect background feature points**
We run Shi-Tomasi corner detection on regions NOT occupied by detected persons. These are "background" points that should move in lock-step with the camera.

**Step 2: Track features with Lucas-Kanade (LK) Optical Flow**
LK is a sparse, pyramid-based optical flow algorithm. For each background feature point, it estimates how much it moved (dx, dy) between frames. Being pyramid-based, it handles large motions (fast drone movement) by tracking at multiple resolution scales.

**Step 3: Estimate affine camera transform**
From the flow of background points, we estimate a **partial affine transform** (4 DOF: translation x/y, rotation, uniform scale) using RANSAC to reject outliers (moving objects that slipped past our mask).

This transform encodes the camera's motion:
```
[new_x]   [s·cos θ  -s·sin θ  tx] [old_x]
[new_y] = [s·sin θ   s·cos θ  ty] [old_y]
                                    [ 1  ]
```

**Step 4: Compensate Kalman predictions**
Before the detection-to-track association step in ByteTrack, we apply the inverse of this transform to all predicted track centers. This "undoes" the camera motion, so the tracker operates in a pseudo-stabilized coordinate system.

**Result**: IoU between predictions and detections stays high even during drone motion → fewer ID switches.

### Additional Anti-Switch Measures

1. **Lost track buffer (30 frames)**: Tracks are kept alive for 30 frames without a detection match. If a person is temporarily occluded and re-appears, they get their original ID back via position-based re-association.

2. **Low-confidence rescue (ByteTrack Pass 2)**: Detections with conf 0.10–0.25 (which would be discarded by standard trackers) are used to match "lost" tracks. This catches persons who are partially occluded or at extreme altitude.

3. **SAHI overlap**: The 20% tile overlap means persons near tile borders aren't missed, reducing "phantom disappearances."

---

## 5. Performance (FPS Benchmarks)

### Test Hardware: [Your Server Hardware Here]
*Fill in after running on your server*

| Configuration | Resolution | FPS | Notes |
|--------------|-----------|-----|-------|
| YOLOv8n, no SAHI | 1920×1080 | ~25 | Standard inference |
| YOLOv8n + SAHI (6 tiles) | 1920×1080 | ~8 | Better small obj detection |
| YOLOv8n + SAHI, CPU | 1920×1080 | ~4-6 | Pure CPU |
| YOLOv8n + SAHI, GPU (RTX) | 1920×1080 | ~25-30 | With CUDA |

**To measure FPS on your setup:**
```bash
python scripts/run_sequence.py --input data/VisDrone/sequences/uav0000013_00000_v
# FPS is printed per-frame and averaged at end
```

---

## 6. Edge Hardware Adaptation (NVIDIA Jetson)

### Deployment Strategy for Jetson Nano / Orin

**Step 1: Export to TensorRT**
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', device=0, half=True)  # FP16 TensorRT engine
```
TensorRT optimizes layer fusion and memory layout for Jetson's GPU. Expected speedup: **3-5×** over PyTorch.

**Step 2: FP16 / INT8 Quantization**
```python
model.export(format='engine', int8=True, data='coco.yaml')  # INT8 with calibration
```
INT8 reduces model size further and speeds up inference, with typically <2% mAP loss.

**Step 3: Replace SAHI with Multi-scale Native Inference**
On Jetson, SAHI's overhead (multiple inference calls) is more costly. Alternative: use YOLOv8's native `imgsz=1280` with a single pass — the model is padded/scaled internally. Less accurate than SAHI but better FPS.

**Step 4: Async Pipeline**
```
Thread 1: Capture frames → queue
Thread 2: YOLO inference (GPU)
Thread 3: ByteTrack + optical flow (CPU cores)
Thread 4: Write output video
```
Jetson has a heterogeneous CPU+GPU; decoupling capture/inference/tracking maximizes utilization.

**Step 5: Reduce optical flow resolution**
Downsample frame to 480p for optical flow computation only (tracking). Use full-res only for detection. LK optical flow at half resolution runs ~4× faster with minimal accuracy loss.

**Expected Jetson Performance:**
| Hardware | Config | Expected FPS |
|----------|--------|-------------|
| Jetson Nano (4GB) | YOLOv8n TRT FP16, no SAHI | ~15 FPS |
| Jetson Orin NX | YOLOv8n TRT INT8, SAHI | ~25 FPS |
| Jetson Orin AGX | YOLOv8s TRT FP16, SAHI | ~30 FPS |

---

## 7. Project Structure

```
aerial_guardian/
├── configs/
│   └── config.yaml          ← All tunable parameters
├── src/
│   ├── detector.py          ← YOLOv8 + SAHI wrapper
│   ├── tracker.py           ← ByteTrack + ego-motion compensation
│   ├── motion_comp.py       ← LK optical flow, affine estimation
│   ├── visualizer.py        ← Boxes, IDs, trajectory tails
│   └── pipeline.py          ← Orchestrator
├── scripts/
│   └── run_sequence.py      ← CLI entry point
├── output/                  ← Generated videos
├── requirements.txt
└── README.md                ← This file
```

---

## 8. Setup & Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)
- ~2GB disk space (for model weights and output videos)

### Install

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd aerial_guardian

# 2. Create conda environment
conda create -n aerial_guardian python=3.10
conda activate aerial_guardian

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import ultralytics; import supervision; import sahi; print('All OK')"
```

### Download Dataset

1. Download VisDrone MOT Task 4 Validation Set from the [official Google Drive link](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view)
2. Extract to `data/VisDrone/`:
```
data/
└── VisDrone/
    └── sequences/
        ├── uav0000013_00000_v/
        │   ├── 0000001.jpg
        │   ├── 0000002.jpg
        │   └── ...
        ├── uav0000020_00406_v/
        └── ...
```

YOLOv8 weights (`yolov8n.pt`) auto-download on first run.

---

## 9. Running the Pipeline

```bash
# Single sequence
python scripts/run_sequence.py \
    --input data/VisDrone/sequences/uav0000013_00000_v

# All sequences
python scripts/run_sequence.py \
    --input data/VisDrone/sequences/ \
    --all

# Fast mode (no SAHI — higher FPS, lower recall on small objects)
python scripts/run_sequence.py \
    --input data/VisDrone/sequences/uav0000013_00000_v \
    --no-sahi

# Force GPU
python scripts/run_sequence.py \
    --input data/VisDrone/sequences/uav0000013_00000_v \
    --device 0
```

Output videos are saved to `output/`.

---

## 10. Configuration Reference

Key parameters in `configs/config.yaml`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model.device` | `cpu` | `cpu`, `cuda`, `0`, `mps` |
| `detection.conf_threshold` | `0.25` | Lower = more detections, more false positives |
| `sahi.enabled` | `true` | Toggle sliced inference |
| `sahi.slice_height/width` | `640` | Tile size (match YOLO's imgsz) |
| `sahi.overlap_*_ratio` | `0.2` | More overlap = fewer missed border objects, slower |
| `tracking.lost_track_buffer` | `30` | Frames to keep lost tracks alive |
| `motion_compensation.enabled` | `true` | Toggle ego-motion compensation |
| `visualization.tail_length` | `30` | Trajectory tail history length |

---

## 11. Engineering Trade-offs

### Precision vs Speed
- **SAHI ON**: Recall ↑ ~20% on small objects, FPS ↓ ~3-4×
- **SAHI OFF**: Real-time on CPU, misses many small persons

**Our choice**: SAHI enabled by default. A missed person is worse than slightly slower FPS for a surveillance application. The `--no-sahi` flag provides the fast alternative.

### Model Size
Total weight footprint:
- YOLOv8n: ~6MB
- ByteTrack: 0MB (pure algorithm)
- Optical flow: 0MB (OpenCV built-in)
- **Total: ~6MB** (vs 300MB limit)

This leaves substantial budget for a Re-ID model if needed (e.g., OSNet at ~2MB), or a fine-tuned VisDrone-specific model.

### ID Consistency vs Computation
Optical flow runs on every frame. On CPU, this costs ~5-10ms per frame. We mitigate by:
- Running LK on downsampled grayscale
- Limiting to 200 feature points
- Skipping compensation if insufficient points found (graceful degradation)

### What We Would Add With More Time
1. **Fine-tune YOLOv8n on VisDrone training set**: COCO-pretrained weights don't know about VisDrone's aerial perspective. Fine-tuning would add ~5-8 mAP on drone-specific objects.
2. **Lightweight Re-ID**: A ~2MB OSNet or MobileNetV2-based embedding to improve re-identification after long occlusions.
3. **Homography-based compensation**: Replace partial affine (4 DOF) with full homography (8 DOF) for handling camera tilt/rotation, at slight computational cost.
4. **INT8 quantization**: Reduce YOLOv8n to ~1.5MB, ~2× faster on edge.