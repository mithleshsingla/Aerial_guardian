#!/usr/bin/env python3
"""
fine_tune.py
============
Fine-tune any YOLO model on VisDrone person detection.

SUPPORTED MODELS (all under 300MB, all real-time capable):
  yolov8n.pt  —  6.2MB,  COCO mAP 37.3, ~84 FPS  (proven baseline)
  yolo11n.pt  —  5.4MB,  COCO mAP 39.5, ~91 FPS  (recommended: better+faster)
  yolov8s.pt  — 21.5MB,  COCO mAP 44.9, ~43 FPS  (accuracy tier)
  yolo11s.pt  — 18.4MB,  COCO mAP 47.0, ~67 FPS  (best accuracy <20MB)

FINE-TUNING HISTORY (on VisDrone MOT train, nc=1):
  Run A: yolov8n, freeze=10, 5ep    → mAP50=0.462 (test)
  Run B: yolov8n, nc=2,      5ep    → mAP50=0.238 (ignore class collapsed)
  Run C: yolov8n, freeze=5,  100ep  → mAP50=0.512@ep17, plateau from ep22
                                       root cause: frozen backbone = bottleneck
  Run D: yolov8n, freeze=0,  37ep   → mAP50=0.555@ep14
                                       problem: val cls_loss overfit (0.0005 lr)
  Run E: yolov8n, freeze=0,  21ep   → mAP50=0.556@ep7, cls_loss stable
                                       label_smoothing=0.1 fixed overfitting
                                       TRACKING MOTA: fine-tuned < baseline
                                       reason: FP explosion on seq4 (dist shift)

LESSONS APPLIED TO THIS SCRIPT:
  - freeze=0     : full fine-tune essential (backbone must adapt to aerial view)
  - lr0=0.0002   : low enough to prevent cls head overfit
  - label_smoothing=0.1 : prevents cls head memorisation
  - close_mosaic=0 : never disable — drone data needs mosaic throughout
  - Recommended: start from existing best.pt checkpoint, not from scratch

RECOMMENDED WORKFLOW:
  # 1. Quick test (5 epochs) — verify no errors
  python scripts/fine_tune.py --data <yaml> --model yolo11n --test-run

  # 2. Full run — YOLO11n (best trade-off, ~3h on RTX 4090)
  python scripts/fine_tune.py --data <yaml> --model yolo11n --device 0

  # 3. Accuracy tier — YOLOv8s with tile=800 (~5h on RTX 4090)
  python scripts/fine_tune.py --data <yaml> --model yolov8s --tile 800 --device 0

  # 4. Continue from existing checkpoint
  python scripts/fine_tune.py --data <yaml> --model yolo11n \\
      --weights runs/finetune/yolo11n_visdrone/weights/best.pt --device 0

  # 5. After training — run full tracking evaluation
  python scripts/run_sequence.py \\
      --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\
      --all --device 0 \\
      --weights runs/finetune/<name>/weights/best.pt
"""

import argparse
from pathlib import Path


MODEL_CONFIGS = {
    "yolov8n": {
        "weights": "yolov8n.pt",
        "size_mb": 6.2,
        "tile":    640,
        "notes":   "proven baseline, 84 FPS",
    },
    "yolo11n": {
        "weights": "yolo11n.pt",
        "size_mb": 5.4,
        "tile":    640,
        "notes":   "recommended — better mAP AND faster than yolov8n",
    },
    "yolov8s": {
        "weights": "yolov8s.pt",
        "size_mb": 21.5,
        "tile":    800,
        "notes":   "accuracy tier — use tile=800 to offset FPS cost",
    },
    "yolo11s": {
        "weights": "yolo11s.pt",
        "size_mb": 18.4,
        "tile":    640,
        "notes":   "best accuracy under 20MB, 67 FPS",
    },
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune YOLO on VisDrone person detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data",     required=True,
                   help="Path to visdrone_person.yaml")
    p.add_argument("--model",    default="yolo11n",
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Model preset (default: yolo11n)")
    p.add_argument("--weights",  default=None,
                   help="Override starting weights (e.g. existing best.pt). "
                        "Defaults to COCO pretrained for chosen --model.")
    p.add_argument("--tile",     type=int, default=None,
                   help="SAHI tile size for training imgsz. Defaults to model preset.")
    p.add_argument("--device",   default="0")
    p.add_argument("--epochs",   type=int, default=80)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--project",  default="runs/finetune")
    p.add_argument("--name",     default=None,
                   help="Run name. Defaults to <model>_visdrone.")
    p.add_argument("--test-run", action="store_true",
                   help="5-epoch smoke test")
    p.add_argument("--resume",   action="store_true",
                   help="Resume from last.pt if interrupted")
    return p.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    preset  = MODEL_CONFIGS[args.model]
    weights = args.weights or preset["weights"]
    tile    = args.tile    or preset["tile"]
    name    = args.name    or f"{args.model}_visdrone"

    if args.test_run:
        args.epochs = 5
        args.batch  = 8

    if args.resume:
        last = Path(args.project) / name / "weights" / "last.pt"
        if last.exists():
            print(f"[FineTune] Resuming from {last}")
            weights = str(last)

    is_warm = weights != preset["weights"]  # starting from checkpoint not COCO

    print(f"\n{'='*60}")
    print(f"  Fine-Tune: {args.model}  ({preset['size_mb']}MB — {preset['notes']})")
    print(f"{'='*60}")
    print(f"  Weights       : {weights}")
    print(f"  Start state   : {'warm checkpoint' if is_warm else 'COCO pretrained'}")
    print(f"  imgsz / tile  : {tile}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch         : {args.batch}")
    print(f"  freeze        : 0  (full fine-tune — backbone must adapt to aerial view)")
    print(f"  lr0           : 0.0002  (prevents cls head overfit — lesson from Run D)")
    print(f"  label_smooth  : 0.1    (fixes val cls_loss divergence — lesson from Run D)")
    print(f"  close_mosaic  : 0      (never disable — drone data needs mosaic always)")
    print(f"  Device        : {args.device}")
    print(f"{'='*60}\n")

    model = YOLO(weights)

    results = model.train(
        data     = args.data,
        epochs   = args.epochs,
        imgsz    = tile,
        batch    = args.batch,
        device   = args.device,
        project  = args.project,
        name     = name,
        exist_ok = True,
        resume   = args.resume,

        # ── Full fine-tune ────────────────────────────────────────────
        freeze   = 0,

        # ── Optimiser ────────────────────────────────────────────────
        optimizer       = "AdamW",
        lr0             = 0.0002,
        lrf             = 0.01,
        warmup_epochs   = 2 if is_warm else 5,
        weight_decay    = 0.001,
        momentum        = 0.937,

        # ── Key regularisation (from Run E lessons) ───────────────────
        label_smoothing = 0.1,

        # ── Drone-specific augmentations ──────────────────────────────
        mosaic          = 1.0,
        copy_paste      = 0.3,
        mixup           = 0.1,
        erasing         = 0.4,
        auto_augment    = "randaugment",

        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.5,    # covers night/dusk sequences

        degrees         = 10.0,
        translate       = 0.1,
        scale           = 0.8,    # 0.2–1.8× simulates altitude changes
        shear           = 2.0,
        perspective     = 0.0005,
        flipud          = 0.3,    # nadir view is top-bottom symmetric
        fliplr          = 0.5,

        close_mosaic    = 0,      # never disable

        patience        = 25,
        workers         = 8,
        cache           = False,
        amp             = True,
        plots           = True,
        verbose         = True,
    )

    # ── Post-training ─────────────────────────────────────────────────
    best = Path(args.project) / name / "weights" / "best.pt"

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights : {best}")

    if best.exists():
        print(f"\n  Validating best weights (IoU=0.3 for tiny aerial objects)...")
        val_model    = YOLO(str(best))
        val_results  = val_model.val(
            data    = args.data,
            imgsz   = tile,
            device  = args.device,
            conf    = 0.35,
            iou     = 0.3,
            verbose = False,
        )
        map50    = val_results.box.map50
        map50_95 = val_results.box.map
        print(f"  Person mAP@0.5 (IoU≥0.3) : {map50:.4f}")
        print(f"  Person mAP@0.5:0.95       : {map50_95:.4f}")
        print(f"\n  Run full tracking evaluation:")
        print(f"    python scripts/run_sequence.py \\")
        print(f"        --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\")
        print(f"        --all --device {args.device} \\")
        print(f"        --weights {best.resolve()}")
        print(f"\n  Compare against baseline:")
        print(f"    python scripts/compare_weights.py \\")
        print(f"        --dataset /home/mithlesh/Object_tracking/VisDrone2019-MOT-val \\")
        print(f"        --baseline yolov8n.pt \\")
        print(f"        --finetuned {best.resolve()} \\")
        print(f"        --device {args.device} --max-frames 150")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()