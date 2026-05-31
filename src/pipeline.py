"""
pipeline.py  —  Aerial Guardian orchestrator
Working version: MOTA@0.5=0.3103, MOTA@0.3=0.4063
"""

import os, sys, glob, time
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List
import supervision as sv

sys.path.insert(0, os.path.dirname(__file__))
from detector          import DroneDetector
from tracker           import AerialTracker
from visualizer        import Visualizer
from annotation_loader import AnnotationLoader
from evaluator         import Evaluator


class AerialGuardianPipeline:

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.detector   = DroneDetector(self.cfg)
        self.tracker    = AerialTracker(self.cfg)
        self.visualizer = Visualizer(self.cfg)
        os.makedirs(self.cfg["io"]["output_dir"], exist_ok=True)

    def _probe_scene_density(self, frame_files: List[str], n_probe: int = 20) -> float:
        counts = []
        for fpath in frame_files[:n_probe]:
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            counts.append(len(self.detector.detect(frame)))
        return float(np.mean(counts)) if counts else 0.0

    def _adapt_conf_for_density(self, avg_dets: float) -> Optional[float]:
        if avg_dets < 2.0:
            return 0.55
        if avg_dets < 6.0:
            return 0.45
        return None

    def _update_detector_conf(self, new_conf: float):
        self.detector.conf = new_conf
        if hasattr(self.detector, "sahi_model") and self.detector.sahi_model is not None:
            self.detector.sahi_model.confidence_threshold = new_conf

    def run_sequence(
        self,
        sequence_name: str,
        dataset_root:  str,
        output_name:   Optional[str] = None,
        evaluate:      bool = True,
        max_frames:    Optional[int] = None,
    ) -> Dict:
        seq_dir  = os.path.join(dataset_root, "sequences",   sequence_name)
        ann_file = os.path.join(dataset_root, "annotations", f"{sequence_name}.txt")

        if not os.path.isdir(seq_dir):
            raise FileNotFoundError(f"Sequence dir not found: {seq_dir}")

        ann_exists = os.path.exists(ann_file)
        if not ann_exists:
            evaluate = False

        print(f"\n{'='*65}")
        print(f"  Sequence : {sequence_name}")
        print(f"  GT Ann   : {'✓' if ann_exists else 'NOT FOUND'}")
        print(f"{'='*65}")

        frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
        if not frame_files:
            raise FileNotFoundError(f"No JPG frames in {seq_dir}")
        if max_frames:
            frame_files = frame_files[:max_frames]
        n_frames = len(frame_files)

        base_conf     = self.cfg["detection"]["conf_threshold"]
        avg_dets      = self._probe_scene_density(frame_files)
        conf_override = self._adapt_conf_for_density(avg_dets)

        if conf_override is not None and conf_override != base_conf:
            print(f"  [Adapt] avg_dets={avg_dets:.1f} → conf {base_conf:.2f} → {conf_override:.2f}")
            self._update_detector_conf(conf_override)
        else:
            print(f"  [Adapt] avg_dets={avg_dets:.1f} → conf={base_conf:.2f} (dense)")
            self._update_detector_conf(base_conf)

        ann_loader = AnnotationLoader(ann_file, person_only=True) if ann_exists else None
        if ann_loader:
            print(ann_loader.summary())

        first = cv2.imread(frame_files[0])
        if first is None:
            raise IOError(f"Cannot read {frame_files[0]}")
        H, W = first.shape[:2]
        print(f"  Resolution : {W} × {H}  |  Frames : {n_frames}")

        if output_name is None:
            output_name = sequence_name
        out_path = os.path.join(self.cfg["io"]["output_dir"], f"{output_name}.mp4")
        writer   = self._make_writer(out_path, self.cfg["io"]["output_fps"], W, H)

        self.tracker.reset()
        evaluator_50 = Evaluator(iou_threshold=0.5) if evaluate else None
        evaluator_30 = Evaluator(iou_threshold=0.3) if evaluate else None
        fps_window: List[float] = []
        total_time = 0.0
        processed_frames = 0
        show_gt = self.cfg["visualization"].get("show_gt_overlay", False)

        for i, fpath in enumerate(frame_files):
            frame_id = int(Path(fpath).stem)
            t0 = time.perf_counter()

            frame = cv2.imread(fpath)
            if frame is None:
                continue

            raw_dets = self.detector.detect(frame)
            sv_dets  = self.detector.detections_to_sv_format(raw_dets, (H, W))
            tracked  = self.tracker.update(sv_dets, frame)

            t1 = time.perf_counter()
            dt = t1 - t0
            fps_window.append(dt)
            total_time += dt
            processed_frames += 1
            if len(fps_window) > 30:
                fps_window.pop(0)
            current_fps = len(fps_window) / sum(fps_window) if fps_window else 0.0

            if ann_loader is not None:
                gt_boxes  = ann_loader.get_active_persons(frame_id)
                pred_xyxy = (tracked.xyxy
                             if tracked.tracker_id is not None and len(tracked) > 0
                             else np.zeros((0, 4), dtype=np.float32))
                pred_ids  = (tracked.tracker_id
                             if tracked.tracker_id is not None and len(tracked) > 0
                             else np.array([], dtype=int))
                if evaluator_50:
                    evaluator_50.update(frame_id, gt_boxes, pred_xyxy, pred_ids)
                if evaluator_30:
                    evaluator_30.update(frame_id, gt_boxes, pred_xyxy, pred_ids)

            gt_viz = (ann_loader.get_active_persons(frame_id)
                      if (show_gt and ann_loader) else [])
            annotated = self.visualizer.draw(
                frame=frame, tracked=tracked,
                track_histories=self.tracker.track_history,
                fps=current_fps, frame_idx=frame_id, gt_boxes=gt_viz,
            )
            writer.write(annotated)

            if (i + 1) % 50 == 0 or i == 0:
                n_t = len(tracked) if tracked.tracker_id is not None else 0
                print(f"  [{i+1:4d}/{n_frames}] frame={frame_id:06d} | "
                      f"det={len(raw_dets):2d} | tracked={n_t:2d} | fps={current_fps:5.1f}")

        writer.release()
        self._update_detector_conf(base_conf)

        avg_fps = processed_frames / total_time if total_time else 0.0
        print(f"\n  ✓ Output  : {out_path}")
        print(f"  ✓ Avg FPS : {avg_fps:.1f}")

        result = {
            "sequence":    sequence_name,
            "output_path": out_path,
            "avg_fps":     round(avg_fps, 2),
            "n_frames":    n_frames,
        }

        if evaluator_50:
            m50 = evaluator_50.compute()
            m30 = evaluator_30.compute()
            print("\n" + evaluator_50.report(sequence_name))
            print(f"  [IoU≥0.3]  MOTA={m30['MOTA']:.4f}  "
                  f"Prec={m30['Precision']:.4f}  Rec={m30['Recall']:.4f}  "
                  f"IDSW={m30['IDSW']}")
            result["metrics"]    = m50
            result["metrics_30"] = m30

        return result

    def run_all_sequences(
        self,
        dataset_root:       str,
        max_frames_per_seq: Optional[int] = None,
        evaluate:           bool = True,
    ) -> List[Dict]:
        seq_base  = os.path.join(dataset_root, "sequences")
        sequences = sorted([
            d for d in os.listdir(seq_base)
            if os.path.isdir(os.path.join(seq_base, d))
        ])
        print(f"\n[Pipeline] {len(sequences)} sequences | "
              f"model={self.cfg['model']['weights']} | "
              f"base_conf={self.cfg['detection']['conf_threshold']}")

        all_results = []
        for seq in sequences:
            try:
                r = self.run_sequence(
                    sequence_name=seq, dataset_root=dataset_root,
                    max_frames=max_frames_per_seq, evaluate=evaluate,
                )
                all_results.append(r)
            except Exception as e:
                print(f"  [ERROR] {seq}: {e}")
                import traceback; traceback.print_exc()

        if all_results:
            self._aggregate_summary(all_results)
        return all_results

    @staticmethod
    def _make_writer(path: str, fps: float, W: int, H: int) -> cv2.VideoWriter:
        for codec in ("mp4v", "avc1", "XVID"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (W, H))
            if w.isOpened():
                return w
        raise RuntimeError(f"Cannot open VideoWriter for {path}")

    def _aggregate_summary(self, results: List[Dict]):
        print(f"\n{'='*65}")
        print(f"  AGGREGATE  ({len(results)} sequences)")
        print(f"{'='*65}")
        fps_vals = [r["avg_fps"] for r in results]
        print(f"  Mean FPS : {np.mean(fps_vals):.1f}  "
              f"(min={min(fps_vals):.1f}  max={max(fps_vals):.1f})")

        for key, label in [("metrics", "IoU≥0.5 (official)"),
                            ("metrics_30", "IoU≥0.3 (practical)")]:
            with_m = [r for r in results if key in r]
            if not with_m:
                continue
            tp = sum(r[key]["TP"]       for r in with_m)
            fp = sum(r[key]["FP"]       for r in with_m)
            fn = sum(r[key]["FN"]       for r in with_m)
            gt = sum(r[key]["GT_total"] for r in with_m)
            sw = sum(r[key]["IDSW"]     for r in with_m)
            mota = 1.0 - (fn + fp + sw) / gt if gt else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec  = tp / (tp + fn) if (tp + fn) else 0
            print(f"\n  [{label}]")
            print(f"  Overall MOTA      : {mota:.4f}")
            print(f"  Overall Precision : {prec:.4f}")
            print(f"  Overall Recall    : {rec:.4f}")
            print(f"  Total ID Switches : {sw}")
            print(f"  Total GT boxes    : {gt}")

        with_m = [r for r in results if "metrics" in r]
        if with_m:
            print(f"\n  Per-sequence  [IoU≥0.5 | IoU≥0.3]:")
            for r in with_m:
                m50 = r["metrics"]
                m30 = r.get("metrics_30", {})
                v50 = m50["MOTA"]
                v30 = m30.get("MOTA", float("nan"))
                sw  = m50["IDSW"]
                icon = "✓" if v50 > 0.35 else ("~" if v50 > 0 else "✗")
                print(f"    {icon} {r['sequence']:<42s}  "
                      f"MOTA={v50:>7.4f} | {v30:>7.4f}  "
                      f"IDSW={sw:>4d}  FPS={r['avg_fps']:>5.1f}")
        print(f"{'='*65}\n")