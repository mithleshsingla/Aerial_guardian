"""
detector.py — Multi-scale batched SAHI detector with WBF.
Batched: all tiles in ONE YOLO call → higher FPS than sequential SAHI.
MOTA fix: wbf_skip_box_threshold must be <= conf_threshold (0.20, not 0.50).
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False
    print("[WARNING] ensemble_boxes not installed — falling back to NMS merge.")

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
    """Batched multi-scale SAHI detector. All tiles in ONE YOLO call."""

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

        self.use_sahi    = sahi_cfg.get("enabled", True)
        self.tile_h      = sahi_cfg["slice_height"]
        self.tile_w      = sahi_cfg["slice_width"]
        self.overlap_h   = sahi_cfg["overlap_height_ratio"]
        self.overlap_w   = sahi_cfg["overlap_width_ratio"]

        self.multi_scale  = sahi_cfg.get("multi_scale", False)
        self.second_tile  = sahi_cfg.get("second_tile_size", 512)
        self.wbf_iou_thr  = sahi_cfg.get("wbf_iou_threshold", 0.5)
        self.wbf_skip_thr = sahi_cfg.get("wbf_skip_box_threshold", 0.20)

        filt = det_cfg.get("filtering", {})
        self.use_filters = filt.get("enabled",    True)
        self.aspect_min  = filt.get("aspect_min", 0.8)
        self.aspect_max  = filt.get("aspect_max", 6.0)
        self.min_area    = filt.get("min_area",   16)

        family = "YOLO11" if "yolo11" in self.weights.lower() else "YOLOv8"
        print(f"[Detector] {family} | {self.weights} | device={self.device}")
        self.model = YOLO(self.weights)

        mode = ("batched multi-scale" if (self.use_sahi and self.multi_scale)
                else "batched single-scale" if self.use_sahi
                else "full-frame")
        scales = (f"({self.second_tile}+{self.tile_w})"
                  if self.multi_scale else str(self.tile_w))
        print(f"[Detector] mode={mode} | tiles={scales} | conf={self.conf:.2f} | "
              f"wbf_skip={self.wbf_skip_thr:.2f} | {'WBF' if WBF_AVAILABLE else 'NMS'}")

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        if not self.use_sahi:
            return self._detect_standard(frame_bgr)
        return self._detect_batched(frame_bgr)

    def _detect_batched(self, frame_bgr: np.ndarray) -> List[Detection]:
        H_orig, W_orig = frame_bgr.shape[:2]

        if W_orig > MAX_SAHI_WIDTH:
            scale   = MAX_SAHI_WIDTH / W_orig
            frame_s = cv2.resize(frame_bgr, (MAX_SAHI_WIDTH, int(H_orig * scale)),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            frame_s = frame_bgr
            scale   = 1.0

        H_s, W_s = frame_s.shape[:2]

        tile_specs = self._generate_tiles(frame_s, self.tile_w, self.tile_h)
        if self.multi_scale:
            tile_specs += self._generate_tiles(frame_s, self.second_tile, self.second_tile)

        if not tile_specs:
            return []

        crops = [spec[0] for spec in tile_specs]

        batch_results = self.model.predict(
            crops,
            conf=self.conf,
            iou=self.iou_thr,
            imgsz=max(self.tile_w, self.tile_h),
            device=self.device,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )

        all_boxes, all_scores, all_labels = [], [], []

        for result, (crop, x_off, y_off, t_w, t_h) in zip(batch_results, tile_specs):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for i in range(len(result.boxes)):
                if int(result.boxes.cls[i]) != self.PERSON_CLASS_ID:
                    continue
                conf_val = float(result.boxes.conf[i])
                if conf_val < self.conf:
                    continue
                bx1, by1, bx2, by2 = result.boxes.xyxy[i].cpu().numpy()
                fx1 = max(0.0, min(1.0, (x_off + bx1) / W_s))
                fy1 = max(0.0, min(1.0, (y_off + by1) / H_s))
                fx2 = max(0.0, min(1.0, (x_off + bx2) / W_s))
                fy2 = max(0.0, min(1.0, (y_off + by2) / H_s))
                all_boxes.append([fx1, fy1, fx2, fy2])
                all_scores.append(conf_val)
                all_labels.append(0)

        if not all_boxes:
            return []

        merged_boxes, merged_scores = self._merge_boxes(all_boxes, all_scores, all_labels)

        detections = []
        for (x1n, y1n, x2n, y2n), conf_val in zip(merged_boxes, merged_scores):
            detections.append(Detection(
                bbox_xyxy=np.array([
                    x1n * W_s / scale, y1n * H_s / scale,
                    x2n * W_s / scale, y2n * H_s / scale,
                ], dtype=np.float32),
                confidence=float(conf_val),
                class_id=self.PERSON_CLASS_ID,
                class_name="person",
            ))
        return self._filter(detections)

    def _generate_tiles(self, frame, tile_w, tile_h):
        H, W = frame.shape[:2]
        overlap_h = min(self.overlap_h + 0.05, 0.35) if W > 1600 else self.overlap_h
        overlap_w = min(self.overlap_w + 0.05, 0.35) if W > 1600 else self.overlap_w
        stride_x  = max(1, int(tile_w * (1 - overlap_w)))
        stride_y  = max(1, int(tile_h * (1 - overlap_h)))
        tiles = []
        y = 0
        while y < H:
            x = 0
            while x < W:
                x2, y2 = min(x + tile_w, W), min(y + tile_h, H)
                crop = frame[y:y2, x:x2]
                if crop.shape[0] < tile_h or crop.shape[1] < tile_w:
                    padded = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                    padded[:crop.shape[0], :crop.shape[1]] = crop
                    crop = padded
                tiles.append((crop, x, y, tile_w, tile_h))
                if x + tile_w >= W:
                    break
                x += stride_x
            if y + tile_h >= H:
                break
            y += stride_y
        return tiles

    def _merge_boxes(self, boxes, scores, labels):
        """
        Two-stage merge:
        1. WBF: fuses near-identical boxes across tiles (uses IoU)
        2. Post-WBF NMS at iou_thr=0.4: suppresses partial duplicate
           detections that WBF missed because IoU was below its threshold.
           SAHI NMM uses IoS (Intersection over Smaller) which is more
           aggressive than IoU — a partial box fully inside a larger box
           gets suppressed even at low IoU. This NMS step replicates that.
        """
        if not boxes:
            return [], []

        # Stage 1: WBF
        if WBF_AVAILABLE:
            fb, fs, _ = weighted_boxes_fusion(
                [boxes], [scores], [labels],
                iou_thr=self.wbf_iou_thr,
                skip_box_thr=self.wbf_skip_thr,
            )
            fb = fb.tolist()
            fs = fs.tolist()
        else:
            fb, fs = boxes, scores

        # Stage 2: post-WBF NMS to suppress partial/border duplicates
        if len(fb) > 1:
            boxes_arr = np.array(fb, dtype=np.float32)
            # cv2.dnn.NMSBoxes needs [x, y, w, h]
            xywh = boxes_arr.copy()
            xywh[:, 2] = boxes_arr[:, 2] - boxes_arr[:, 0]
            xywh[:, 3] = boxes_arr[:, 3] - boxes_arr[:, 1]
            keep = cv2.dnn.NMSBoxes(
                xywh.tolist(), fs,
                score_threshold=0.0,
                nms_threshold=0.4,
            )
            if len(keep) > 0:
                keep = keep.flatten()
                fb = [fb[i] for i in keep]
                fs = [fs[i] for i in keep]

        return fb, fs

    def _detect_standard(self, frame_bgr):
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

    def _filter(self, detections):
        if not self.use_filters:
            return detections
        return [d for d in detections
                if d.area >= self.min_area
                and self.aspect_min <= d.aspect_ratio <= self.aspect_max]

    def detections_to_sv_format(self, detections, frame_shape):
        import supervision as sv
        if not detections:
            return sv.Detections.empty()
        xyxy       = np.array([d.bbox_xyxy  for d in detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id   = np.array([d.class_id   for d in detections], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)