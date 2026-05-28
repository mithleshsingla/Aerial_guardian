#!/usr/bin/env python3
"""
run_sequence.py  —  Aerial Guardian CLI
========================================
Usage:
  # Baseline (yolov8n, conf=0.35)
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0

  # YOLO11n (recommended upgrade — better mAP, slightly faster)
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0 --model yolo11n

  # YOLOv8s with tile=800 (accuracy tier)
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0 --model yolov8s

  # Fine-tuned weights
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0 \\
      --weights runs/finetune/yolo11n_visdrone/weights/best.pt

  # Single sequence with GT overlay
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --sequence uav0000086_00000_v --device 0 --show-gt

  # Quick test (first 100 frames)
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0 --max-frames 100

  # Benchmark all models (100 frames each)
  python scripts/benchmark_models.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --device 0 --max-frames 100
"""

import argparse
import os
import sys
import tempfile
import yaml

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR     = os.path.join(PROJECT_DIR, "src")
DEFAULT_CFG = os.path.join(PROJECT_DIR, "configs", "config.yaml")
sys.path.insert(0, SRC_DIR)

# Model presets: weights + tile size
MODEL_PRESETS = {
    "yolov8n": {"weights": "yolov8n.pt", "tile": 640},
    "yolo11n": {"weights": "yolo11n.pt", "tile": 640},
    "yolov8s": {"weights": "yolov8s.pt", "tile": 800},
    "yolo11s": {"weights": "yolo11s.pt", "tile": 640},
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Aerial Guardian — drone person detection & tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset",    required=True,
                   help="Path to VisDrone2019-MOT-val/")
    p.add_argument("--sequence",   default=None,
                   help="Single sequence name")
    p.add_argument("--all",        action="store_true",
                   help="Process all sequences")

    # Model selection — mutually useful (--model sets weights+tile, --weights overrides)
    p.add_argument("--model",    default=None,
                   choices=list(MODEL_PRESETS.keys()),
                   help="Model preset: sets weights + tile size automatically")
    p.add_argument("--weights",  default=None,
                   help="Direct path to .pt weights (overrides --model weights)")
    p.add_argument("--tile",     type=int, default=None,
                   help="Override SAHI tile size (default from --model preset)")

    p.add_argument("--config",     default=DEFAULT_CFG)
    p.add_argument("--device",     default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--conf",       type=float, default=None,
                   help="Override detection confidence threshold")
    p.add_argument("--no-sahi",    action="store_true",
                   help="Disable SAHI (faster, lower recall)")
    p.add_argument("--no-eval",    action="store_true",
                   help="Skip GT evaluation")
    p.add_argument("--show-gt",    action="store_true",
                   help="Draw GT boxes in output video")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.sequence and not args.all:
        print("[ERROR] Provide --sequence <name> or --all")
        sys.exit(1)

    if not os.path.isdir(args.dataset):
        print(f"[ERROR] Dataset not found: {args.dataset}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply model preset first, then individual overrides
    if args.model:
        preset = MODEL_PRESETS[args.model]
        cfg["model"]["weights"]       = preset["weights"]
        cfg["sahi"]["slice_height"]   = preset["tile"]
        cfg["sahi"]["slice_width"]    = preset["tile"]
        print(f"[Config] Model preset → {args.model}  "
              f"(weights={preset['weights']}, tile={preset['tile']})")

    if args.weights:
        cfg["model"]["weights"] = args.weights
        print(f"[Config] Weights  → {args.weights}")

    if args.tile:
        cfg["sahi"]["slice_height"] = args.tile
        cfg["sahi"]["slice_width"]  = args.tile
        print(f"[Config] Tile     → {args.tile}")

    if args.device:
        cfg["model"]["device"] = args.device
        print(f"[Config] Device   → {args.device}")

    if args.conf is not None:
        cfg["detection"]["conf_threshold"] = args.conf
        cfg["tracking"]["track_activation_threshold"] = args.conf
        print(f"[Config] Conf     → {args.conf}")

    if args.no_sahi:
        cfg["sahi"]["enabled"] = False
        print("[Config] SAHI disabled")

    if args.show_gt:
        cfg["visualization"]["show_gt_overlay"] = True

    if args.output_dir:
        cfg["io"]["output_dir"] = args.output_dir

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    ) as tf:
        yaml.dump(cfg, tf)
        tmp = tf.name

    try:
        from pipeline import AerialGuardianPipeline
        p = AerialGuardianPipeline(tmp)

        if args.all:
            p.run_all_sequences(
                dataset_root=args.dataset,
                max_frames_per_seq=args.max_frames,
                evaluate=not args.no_eval,
            )
        else:
            p.run_sequence(
                sequence_name=args.sequence,
                dataset_root=args.dataset,
                max_frames=args.max_frames,
                evaluate=not args.no_eval,
            )
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    main()