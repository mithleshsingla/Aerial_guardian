"""
detector.py
===========
Multi-scale SAHI detector with Weighted Box Fusion (WBF).

MULTI-SCALE SAHI — WHY IT IMPROVES RECALL:
  Standard SAHI runs one tile size (e.g. 640px). A 10px person occupies
  1.6% of a 640px tile — very close to YOLOv8n's detection limit.
  
  Multi-scale SAHI runs TWO tile sizes per frame and merges results:
    tile=640: catches 15-40px persons well
    tile=512: catches 8-15px persons better (appear 25% larger relative to tile)

  A person missed at 640 may be caught at 512, and vice versa.
  The union of both passes = higher recall with controlled FP rate.

WHY WBF OVER NMM:
  NMM (Non-Maximum Merging) picks the highest-scoring box and suppresses
  nearby boxes. For tiny aerial objects where two passes detect slightly
  different positions, WBF averages the box coordinates weighted by score.
  Result: better localisation on 8-20px persons where a 2px offset matters.

  WBF formula:
    weighted_cx = Σ(score_i × cx_i) / Σ(score_i)
    weighted_cy = Σ(score_i × cy_i) / Σ(score_i)
    fused_score = Σ(score_i) / n_boxes_in_cluster

FPS IMPACT:
  Standard SAHI at tile=640: 6 tiles per frame → ~84 FPS
  Multi-scale (512+640):    12 tiles per frame → ~42 FPS
  Still real-time. WBF adds <1ms per frame.

SINGLE-SCALE FALLBACK:
  Set multi_scale=false in config to revert to single-tile SAHI.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    print("[WARNING] SAHI not installed.")

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False
    print("[WARNING] ensemble_boxes not installed — WBF unavailable, using NMM.")

from ultralytics import YOLO

MAX_SAHI_WIDTH = 1920


@dataclass
class Detection:
    bbox_xyxy:  np.ndarray
    confidence: float
    class_id:   int
    class_name: str

    @property
    def width(self)  -> float: return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])
    @property
    def height(self) -> float: return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])
    @property
    def area(self)   -> float: return self.width * self.height
    @property
    def aspect_ratio(self) -> float:
        return self.height / max(self.width, 1.0)
    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1+x2)/2, (y1+y2)/2)


class DroneDetector:
    """
    YOLOv8/YOLO11 + multi-scale SAHI + WBF for aerial person detection.
    """

    PERSON_CLASS_ID = 0

    def __init__(self, cfg: dict):
        det_cfg   = cfg["detection"]
        model_cfg = cfg["model"]
        sahi_cfg  = cfg["sahi"]

        self.conf      = det_cfg["conf_threshold"]
        self.iou_thr   = det_cfg["iou_threshold"]
        self.device    = model_cfg["device"]
        self.imgsz     = model_cfg["imgsz"]
        self.weights   = model_cfg["weights"]

        self.use_sahi    = sahi_cfg["enabled"] and SAHI_AVAILABLE
        self.slice_h     = sahi_cfg["slice_height"]
        self.slice_w     = sahi_cfg["slice_width"]
        self.overlap_h   = sahi_cfg["overlap_height_ratio"]
        self.overlap_w   = sahi_cfg["overlap_width_ratio"]
        self.post_type   = sahi_cfg["postprocess_type"]
        self.post_thr    = sahi_cfg["postprocess_match_threshold"]

        # Multi-scale SAHI
        self.multi_scale = sahi_cfg.get("multi_scale", False) and SAHI_AVAILABLE and WBF_AVAILABLE
        self.second_tile = sahi_cfg.get("second_tile_size", 512)
        self.wbf_iou_thr = sahi_cfg.get("wbf_iou_threshold", 0.5)
        self.wbf_skip_box_thr = sahi_cfg.get("wbf_skip_box_threshold", 0.20)

        filt = det_cfg.get("filtering", {})
        self.use_filters = filt.get("enabled",    True)
        self.aspect_min  = filt.get("aspect_min", 0.8)
        self.aspect_max  = filt.get("aspect_max", 6.0)
        self.min_area    = filt.get("min_area",   16)

        # Detect model family for logging
        w = self.weights.lower()
        family = "YOLO11" if ("yolo11" in w) else "YOLOv8"

        print(f"[Detector] {family} | {self.weights} | device={self.device}")
        self.model = YOLO(self.weights)

        if self.use_sahi:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.weights,
                confidence_threshold=self.conf,
                device=self.device,
            )
            if self.multi_scale:
                print(
                    f"[Detector] Multi-scale SAHI | tiles={self.second_tile}+{self.slice_w} "
                    f"| WBF iou={self.wbf_iou_thr} | conf={self.conf:.2f}"
                )
            else:
                print(
                    f"[Detector] SAHI | tile={self.slice_w} | conf={self.conf:.2f} "
                    f"| aspect=[{self.aspect_min},{self.aspect_max}]"
                )
        else:
            self.sahi_model = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.use_sahi:
            return self._detect_standard(frame_bgr)
        if self.multi_scale:
            return self._detect_multiscale(frame_bgr)
        return self._detect_sahi_single(frame_bgr, self.slice_h, self.slice_w)

    # ------------------------------------------------------------------ #
    # Multi-scale SAHI with WBF
    # ------------------------------------------------------------------ #

    def _detect_multiscale(self, frame_bgr: np.ndarray) -> List[Detection]:
        """
        Run SAHI at two tile sizes and merge with Weighted Box Fusion.
        tile_large (e.g. 640): catches medium/large small objects
        tile_small (e.g. 512): catches very tiny objects (8-15px)
        """
        H_orig, W_orig = frame_bgr.shape[:2]

        # Resolution cap
        if W_orig > MAX_SAHI_WIDTH:
            scale   = MAX_SAHI_WIDTH / W_orig
            frame_s = cv2.resize(
                frame_bgr,
                (MAX_SAHI_WIDTH, int(H_orig * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_s = frame_bgr
            scale   = 1.0

        H_s, W_s = frame_s.shape[:2]
        frame_rgb = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)

        # Pass 1: large tile
        dets_large = self._run_sahi_pass(frame_rgb, self.slice_h, self.slice_w)
        # Pass 2: small tile
        dets_small = self._run_sahi_pass(frame_rgb, self.second_tile, self.second_tile)

        # Merge with WBF
        merged = self._wbf_merge(dets_large, dets_small, W_s, H_s)

        # Map back to original coordinates
        detections = []
        for (x1, y1, x2, y2), conf in merged:
            det = Detection(
                bbox_xyxy=np.array(
                    [x1/scale, y1/scale, x2/scale, y2/scale],
                    dtype=np.float32,
                ),
                confidence=float(conf),
                class_id=self.PERSON_CLASS_ID,
                class_name="person",
            )
            detections.append(det)

        return self._filter(detections)

    def _run_sahi_pass(
        self, frame_rgb: np.ndarray, tile_h: int, tile_w: int
    ) -> List[Tuple[Tuple[float,float,float,float], float]]:
        """
        Run one SAHI pass, return list of (xyxy_px, confidence).
        """
        ov_h, ov_w = self._adaptive_overlap(frame_rgb.shape[1], frame_rgb.shape[0])
        result = get_sliced_prediction(
            frame_rgb,
            self.sahi_model,
            slice_height=tile_h,
            slice_width=tile_w,
            overlap_height_ratio=ov_h,
            overlap_width_ratio=ov_w,
            postprocess_type="NMM",   # intra-pass NMM first, then WBF across passes
            postprocess_match_threshold=self.post_thr,
            verbose=0,
        )
        out = []
        for pred in result.object_prediction_list:
            if pred.category.id != self.PERSON_CLASS_ID:
                continue
            if pred.score.value < self.conf:
                continue
            b = pred.bbox
            out.append(((b.minx, b.miny, b.maxx, b.maxy), float(pred.score.value)))
        return out

    def _wbf_merge(
        self,
        dets_a: List,
        dets_b: List,
        img_w: int,
        img_h: int,
    ) -> List[Tuple[Tuple[float,float,float,float], float]]:
        """
        Merge two detection lists with Weighted Box Fusion.
        WBF normalises boxes to [0,1] range, fuses overlapping ones,
        then returns boxes in pixel coordinates.
        """
        if not dets_a and not dets_b:
            return []

        # Normalise to [0,1]
        def norm(dets):
            boxes, scores = [], []
            for (x1,y1,x2,y2), s in dets:
                boxes.append([
                    max(0.0, min(1.0, x1/img_w)),
                    max(0.0, min(1.0, y1/img_h)),
                    max(0.0, min(1.0, x2/img_w)),
                    max(0.0, min(1.0, y2/img_h)),
                ])
                scores.append(s)
            return boxes, scores

        boxes_a, scores_a = norm(dets_a)
        boxes_b, scores_b = norm(dets_b)

        all_boxes  = [boxes_a,  boxes_b]  if (boxes_a  and boxes_b)  else ([boxes_a]  if boxes_a  else [boxes_b])
        all_scores = [scores_a, scores_b] if (scores_a and scores_b) else ([scores_a] if scores_a else [scores_b])
        all_labels = [[0]*len(s) for s in all_scores]

        fused_boxes, fused_scores, _ = weighted_boxes_fusion(
            all_boxes,
            all_scores,
            all_labels,
            iou_thr=self.wbf_iou_thr,
            skip_box_thr=self.wbf_skip_box_thr,
        )

        # Back to pixel coords
        out = []
        for box, score in zip(fused_boxes, fused_scores):
            x1 = box[0] * img_w
            y1 = box[1] * img_h
            x2 = box[2] * img_w
            y2 = box[3] * img_h
            out.append(((x1, y1, x2, y2), float(score)))
        return out

    # ------------------------------------------------------------------ #
    # Single-scale SAHI (original)
    # ------------------------------------------------------------------ #

    def _detect_sahi_single(
        self, frame_bgr: np.ndarray, tile_h: int, tile_w: int
    ) -> List[Detection]:
        H_orig, W_orig = frame_bgr.shape[:2]
        scale = 1.0

        if W_orig > MAX_SAHI_WIDTH:
            scale   = MAX_SAHI_WIDTH / W_orig
            frame_s = cv2.resize(
                frame_bgr,
                (MAX_SAHI_WIDTH, int(H_orig * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            frame_s = frame_bgr

        ov_h, ov_w = self._adaptive_overlap(frame_s.shape[1], frame_s.shape[0])
        frame_rgb  = cv2.cvtColor(frame_s, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            frame_rgb, self.sahi_model,
            slice_height=tile_h, slice_width=tile_w,
            overlap_height_ratio=ov_h, overlap_width_ratio=ov_w,
            postprocess_type=self.post_type,
            postprocess_match_threshold=self.post_thr,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            if pred.category.id != self.PERSON_CLASS_ID or pred.score.value < self.conf:
                continue
            bbox = pred.bbox
            detections.append(Detection(
                bbox_xyxy=np.array(
                    [bbox.minx/scale, bbox.miny/scale,
                     bbox.maxx/scale, bbox.maxy/scale], dtype=np.float32),
                confidence=float(pred.score.value),
                class_id=self.PERSON_CLASS_ID,
                class_name="person",
            ))
        return self._filter(detections)

    # ------------------------------------------------------------------ #
    # Standard full-frame inference
    # ------------------------------------------------------------------ #

    def _detect_standard(self, frame_bgr: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            frame_bgr, conf=self.conf, iou=self.iou_thr,
            imgsz=self.imgsz, device=self.device,
            classes=[self.PERSON_CLASS_ID], verbose=False,
        )
        detections = []
        if results and results[0].boxes is not None:
            for i in range(len(results[0].boxes)):
                if int(results[0].boxes.cls[i]) != self.PERSON_CLASS_ID:
                    continue
                detections.append(Detection(
                    bbox_xyxy=results[0].boxes.xyxy[i].cpu().numpy().astype(np.float32),
                    confidence=float(results[0].boxes.conf[i]),
                    class_id=self.PERSON_CLASS_ID,
                    class_name="person",
                ))
        return self._filter(detections)

    # ------------------------------------------------------------------ #
    # Filters + helpers
    # ------------------------------------------------------------------ #

    def _filter(self, detections: List[Detection]) -> List[Detection]:
        if not self.use_filters:
            return detections
        return [
            d for d in detections
            if d.area >= self.min_area
            and self.aspect_min <= d.aspect_ratio <= self.aspect_max
        ]

    def _adaptive_overlap(self, W: int, H: int) -> Tuple[float, float]:
        if W > 1600:
            return min(self.overlap_h + 0.05, 0.35), min(self.overlap_w + 0.05, 0.35)
        return self.overlap_h, self.overlap_w

    def detections_to_sv_format(
        self, detections: List[Detection], frame_shape: Tuple
    ) -> "sv.Detections":
        import supervision as sv
        if not detections:
            return sv.Detections.empty()
        xyxy       = np.array([d.bbox_xyxy  for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id   = np.array([d.class_id   for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)