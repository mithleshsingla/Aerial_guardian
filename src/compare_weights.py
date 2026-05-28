#!/usr/bin/env python3
"""
compare_weights.py
==================
Run the full tracking pipeline with two different weight files
and print a side-by-side MOTA comparison table.

Useful to confirm fine-tuned weights beat the COCO baseline.

Usage:
  python scripts/compare_weights.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --baseline yolov8n.pt \\
      --finetuned runs/finetune/visdrone_person/weights/best.pt \\
      --device 0 \\
      --max-frames 150    # quick comparison on first 150 frames
"""

import argparse
import os
import sys
import tempfile
import yaml
import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
SRC_DIR     = os.path.join(PROJECT_DIR, "src")
DEFAULT_CFG = os.path.join(PROJECT_DIR, "configs", "config.yaml")
sys.path.insert(0, SRC_DIR)


def run_with_weights(weights, cfg_path, dataset, device, max_frames, tag):
    """Run pipeline with given weights, return results list."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["weights"] = weights
    cfg["model"]["device"]  = device
    cfg["io"]["output_dir"] = f"output/{tag}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False, dir="/tmp") as tf:
        yaml.dump(cfg, tf)
        tmp = tf.name

    try:
        from pipeline import AerialGuardianPipeline
        p = AerialGuardianPipeline(tmp)
        return p.run_all_sequences(
            dataset_root=dataset,
            max_frames_per_seq=max_frames,
            evaluate=True,
        )
    finally:
        os.unlink(tmp)


def comparison_table(results_a, label_a, results_b, label_b):
    """Print side-by-side per-sequence MOTA comparison."""
    def by_seq(results):
        return {r["sequence"]: r for r in results}

    ra = by_seq(results_a)
    rb = by_seq(results_b)
    seqs = sorted(set(list(ra.keys()) + list(rb.keys())))

    col = 42
    print(f"\n{'='*(col+50)}")
    print(f"  {'Sequence':<{col}}  {label_a:>10}  {label_b:>10}  {'Δ MOTA':>8}")
    print(f"  {'-'*col}  {'-'*10}  {'-'*10}  {'-'*8}")

    deltas = []
    for seq in seqs:
        ma = ra[seq]["metrics"]["MOTA"] if seq in ra and "metrics" in ra[seq] else float("nan")
        mb = rb[seq]["metrics"]["MOTA"] if seq in rb and "metrics" in rb[seq] else float("nan")
        d  = mb - ma if not (np.isnan(ma) or np.isnan(mb)) else float("nan")
        arrow = "↑" if d > 0.01 else ("↓" if d < -0.01 else "≈")
        print(f"  {seq:<{col}}  {ma:>10.4f}  {mb:>10.4f}  {d:>+7.4f} {arrow}")
        if not np.isnan(d):
            deltas.append(d)

    print(f"  {'-'*col}  {'-'*10}  {'-'*10}  {'-'*8}")

    def agg(results):
        with_m = [r for r in results if "metrics" in r]
        if not with_m: return float("nan"), float("nan"), float("nan")
        tp = sum(r["metrics"]["TP"]       for r in with_m)
        fp = sum(r["metrics"]["FP"]       for r in with_m)
        fn = sum(r["metrics"]["FN"]       for r in with_m)
        gt = sum(r["metrics"]["GT_total"] for r in with_m)
        sw = sum(r["metrics"]["IDSW"]     for r in with_m)
        mota = 1.0-(fn+fp+sw)/gt if gt else 0
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec  = tp/(tp+fn) if (tp+fn) else 0
        return mota, prec, rec

    ma_agg, pa, ra_ = agg(results_a)
    mb_agg, pb, rb_ = agg(results_b)
    print(f"  {'OVERALL MOTA':<{col}}  {ma_agg:>10.4f}  {mb_agg:>10.4f}  {mb_agg-ma_agg:>+7.4f}")
    print(f"  {'OVERALL Precision':<{col}}  {pa:>10.4f}  {pb:>10.4f}  {pb-pa:>+7.4f}")
    print(f"  {'OVERALL Recall':<{col}}  {ra_:>10.4f}  {rb_:>10.4f}  {rb_-ra_:>+7.4f}")
    print(f"{'='*(col+50)}\n")

    sw_a = sum(r["metrics"]["IDSW"] for r in results_a if "metrics" in r)
    sw_b = sum(r["metrics"]["IDSW"] for r in results_b if "metrics" in r)
    print(f"  Total ID switches:  {label_a}={sw_a}   {label_b}={sw_b}   Δ={sw_b-sw_a:+d}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--baseline",  default="yolov8n.pt")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--config",    default=DEFAULT_CFG)
    ap.add_argument("--device",    default="0")
    ap.add_argument("--max-frames",type=int, default=None)
    args = ap.parse_args()

    print(f"\n[Compare] Baseline  : {args.baseline}")
    print(f"[Compare] Fine-tuned: {args.finetuned}")
    print(f"[Compare] Max frames: {args.max_frames or 'all'}\n")

    print("── Running baseline ──────────────────────────────────────")
    res_base = run_with_weights(
        args.baseline, args.config, args.dataset,
        args.device, args.max_frames, "baseline"
    )

    print("\n── Running fine-tuned ────────────────────────────────────")
    res_ft = run_with_weights(
        args.finetuned, args.config, args.dataset,
        args.device, args.max_frames, "finetuned"
    )

    comparison_table(res_base, "Baseline", res_ft, "Fine-tuned")


if __name__ == "__main__":
    main()