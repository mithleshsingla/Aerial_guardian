#!/usr/bin/env python3
"""
benchmark_models.py
===================
Compare multiple YOLO models on a VisDrone sequence:
FPS, detection count, mAP@0.5, MOTA@0.5, MOTA@0.3.

Runs each model configuration through the full pipeline on the first
N frames of each sequence and produces a comparison table.

Usage:
  # Quick benchmark (100 frames per sequence)
  python scripts/benchmark_models.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --device 0 --max-frames 100

  # Full benchmark
  python scripts/benchmark_models.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --device 0

  # Include fine-tuned weights
  python scripts/benchmark_models.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --device 0 --max-frames 100 \\
      --extra-weights runs/finetune/yolo11n_visdrone/weights/best.pt
"""

import argparse
import os
import sys
import tempfile
import yaml
import numpy as np
from pathlib import Path

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR     = os.path.join(PROJECT_DIR, "src")
DEFAULT_CFG = os.path.join(PROJECT_DIR, "configs", "config.yaml")
sys.path.insert(0, SRC_DIR)


# Model configurations to benchmark
BENCHMARK_CONFIGS = [
    {
        "label":      "yolov8n (baseline)",
        "weights":    "yolov8n.pt",
        "tile":       640,
        "conf":       0.35,
        "size_mb":    6.2,
    },
    {
        "label":      "yolo11n (recommended)",
        "weights":    "yolo11n.pt",
        "tile":       640,
        "conf":       0.35,
        "size_mb":    5.4,
    },
    {
        "label":      "yolov8s + tile=800",
        "weights":    "yolov8s.pt",
        "tile":       800,
        "conf":       0.35,
        "size_mb":    21.5,
    },
    {
        "label":      "yolo11s",
        "weights":    "yolo11s.pt",
        "tile":       640,
        "conf":       0.35,
        "size_mb":    18.4,
    },
]


def run_config(cfg_dict: dict, dataset: str, device: str,
               max_frames: int, config_path: str) -> dict:
    """Run pipeline with one model config, return aggregate metrics."""
    import copy
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["model"]["weights"]             = cfg_dict["weights"]
    cfg["model"]["device"]              = device
    cfg["sahi"]["slice_height"]         = cfg_dict["tile"]
    cfg["sahi"]["slice_width"]          = cfg_dict["tile"]
    cfg["detection"]["conf_threshold"]  = cfg_dict["conf"]
    cfg["tracking"]["track_activation_threshold"] = cfg_dict["conf"]
    cfg["io"]["output_dir"]             = f"output/benchmark_{cfg_dict['label'].replace(' ','_')}"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="/tmp"
    ) as tf:
        yaml.dump(cfg, tf)
        tmp = tf.name

    try:
        from pipeline import AerialGuardianPipeline
        p       = AerialGuardianPipeline(tmp)
        results = p.run_all_sequences(
            dataset_root=dataset,
            max_frames_per_seq=max_frames,
            evaluate=True,
        )
    finally:
        os.unlink(tmp)

    # Aggregate
    with_m  = [r for r in results if "metrics" in r]
    with_m3 = [r for r in results if "metrics_30" in r]

    def agg(key):
        rs = [r for r in results if key in r]
        if not rs: return {}
        tp = sum(r[key]["TP"]       for r in rs)
        fp = sum(r[key]["FP"]       for r in rs)
        fn = sum(r[key]["FN"]       for r in rs)
        gt = sum(r[key]["GT_total"] for r in rs)
        sw = sum(r[key]["IDSW"]     for r in rs)
        return {
            "MOTA":  round(1.0-(fn+fp+sw)/gt, 4) if gt else 0,
            "Prec":  round(tp/(tp+fp), 4) if (tp+fp) else 0,
            "Rec":   round(tp/(tp+fn), 4) if (tp+fn) else 0,
            "IDSW":  sw,
        }

    fps_vals = [r["avg_fps"] for r in results]
    return {
        "label":    cfg_dict["label"],
        "size_mb":  cfg_dict["size_mb"],
        "tile":     cfg_dict["tile"],
        "conf":     cfg_dict["conf"],
        "avg_fps":  round(float(np.mean(fps_vals)), 1),
        "m50":      agg("metrics"),
        "m30":      agg("metrics_30"),
    }


def print_table(all_results: list):
    col = 28
    print(f"\n{'='*95}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*95}")
    print(f"  {'Model':<{col}}  {'Size':>6}  {'Tile':>5}  "
          f"{'FPS':>6}  {'MOTA@0.5':>9}  {'MOTA@0.3':>9}  "
          f"{'Prec':>6}  {'Rec':>6}  {'IDSW':>5}")
    print(f"  {'-'*col}  {'-'*6}  {'-'*5}  "
          f"{'-'*6}  {'-'*9}  {'-'*9}  "
          f"{'-'*6}  {'-'*6}  {'-'*5}")

    for r in all_results:
        m50 = r.get("m50", {})
        m30 = r.get("m30", {})
        mota50 = m50.get("MOTA", float("nan"))
        mota30 = m30.get("MOTA", float("nan"))
        prec   = m50.get("Prec", float("nan"))
        rec    = m50.get("Rec",  float("nan"))
        idsw   = m50.get("IDSW", "-")
        icon   = "★" if mota50 == max(
            x.get("m50",{}).get("MOTA", -999) for x in all_results
        ) else " "
        print(
            f"  {icon}{r['label']:<{col-1}}  "
            f"{r['size_mb']:>5.1f}M  "
            f"{r['tile']:>5}  "
            f"{r['avg_fps']:>6.1f}  "
            f"{mota50:>9.4f}  "
            f"{mota30:>9.4f}  "
            f"{prec:>6.3f}  "
            f"{rec:>6.3f}  "
            f"{idsw:>5}"
        )
    print(f"{'='*95}\n")
    print("  ★ = best MOTA@0.5")
    print("  All models: aspect filter h/w∈[0.8,6.0], SAHI NMM, homography compensation")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark YOLO model variants on VisDrone tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dataset",       required=True)
    ap.add_argument("--config",        default=DEFAULT_CFG)
    ap.add_argument("--device",        default="0")
    ap.add_argument("--max-frames",    type=int, default=None)
    ap.add_argument("--extra-weights", nargs="*", default=[],
                    help="Additional fine-tuned .pt files to include")
    ap.add_argument("--models",        nargs="*",
                    default=["yolov8n", "yolo11n", "yolov8s", "yolo11s"],
                    help="Which presets to run (default: all four)")
    args = ap.parse_args()

    configs = [c for c in BENCHMARK_CONFIGS
               if any(m in c["label"] for m in args.models)]

    # Add extra fine-tuned weights
    for w in args.extra_weights:
        name = Path(w).parent.parent.name  # runs/finetune/<name>/weights/best.pt
        configs.append({
            "label":   f"fine-tuned ({name})",
            "weights": w,
            "tile":    640,
            "conf":    0.35,
            "size_mb": Path(w).stat().st_size / 1024 / 1024 if Path(w).exists() else 0,
        })

    print(f"\n[Benchmark] {len(configs)} configs × "
          f"{'all frames' if args.max_frames is None else str(args.max_frames)+' frames/seq'}")

    all_results = []
    for cfg in configs:
        print(f"\n── {cfg['label']} ──────────────────────────────")
        try:
            r = run_config(cfg, args.dataset, args.device,
                           args.max_frames, args.config)
            all_results.append(r)
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback; traceback.print_exc()

    if all_results:
        print_table(all_results)


if __name__ == "__main__":
    main()