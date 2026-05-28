"""
visualizer.py — draws boxes, IDs, trajectory tails, and GT overlays.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import supervision as sv

PALETTE = [
    (57, 255, 20),   (255, 69, 0),   (30, 144, 255),  (255, 215, 0),
    (148, 0, 211),   (0, 255, 255),  (255, 20, 147),  (127, 255, 0),
    (255, 140, 0),   (0, 191, 255),  (255, 0, 128),   (50, 205, 50),
    (220, 20, 60),   (64, 224, 208), (255, 165, 0),   (138, 43, 226),
]

def get_color(tid: int) -> Tuple[int,int,int]:
    return PALETTE[int(tid) % len(PALETTE)]


class Visualizer:
    def __init__(self, cfg: dict):
        v = cfg["visualization"]
        self.tail_length  = v["tail_length"]
        self.tail_thick   = v["tail_thickness"]
        self.box_thick    = v["box_thickness"]
        self.text_scale   = v["text_scale"]
        self.text_thick   = v["text_thickness"]
        self.show_fps     = cfg["io"]["show_fps_overlay"]

    def draw(
        self,
        frame: np.ndarray,
        tracked: "sv.Detections",
        track_histories: Dict[int, List[Tuple[float,float]]],
        fps: Optional[float] = None,
        frame_idx: Optional[int] = None,
        gt_boxes=None,          # list of GTBox, optional
    ) -> np.ndarray:
        canvas = frame.copy()

        # GT overlay (dashed yellow boxes for debugging)
        if gt_boxes:
            self.draw_gt_overlay(canvas, gt_boxes)

        if tracked.tracker_id is None or len(tracked) == 0:
            self._draw_fps(canvas, fps, frame_idx)
            return canvas

        # Tails (draw first, behind boxes)
        for tid in tracked.tracker_id:
            hist = track_histories.get(int(tid), [])
            if len(hist) >= 2:
                self._draw_tail(canvas, hist, int(tid))

        # Boxes + labels
        for i, tid in enumerate(tracked.tracker_id):
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            color = get_color(int(tid))
            cv2.rectangle(canvas, (x1,y1), (x2,y2), color, self.box_thick)

            label = f"P#{int(tid)} {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.text_scale, self.text_thick)
            ly1 = max(y1 - lh - bl - 4, 0)
            cv2.rectangle(canvas, (x1, ly1), (x1+lw+4, y1), color, -1)
            cv2.putText(canvas, label, (x1+2, y1-bl-2),
                        cv2.FONT_HERSHEY_SIMPLEX, self.text_scale,
                        (0,0,0), self.text_thick, cv2.LINE_AA)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(canvas, (cx,cy), 3, color, -1)

        # Count
        cv2.putText(canvas, f"Persons: {len(tracked)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        self._draw_fps(canvas, fps, frame_idx)
        return canvas

    def _draw_tail(self, canvas, history, tid):
        color = get_color(tid)
        n = len(history)
        for i in range(1, n):
            alpha = i / n
            t = max(1, int(self.tail_thick * alpha))
            faded = tuple(int(c * alpha) for c in color)
            p1 = (int(history[i-1][0]), int(history[i-1][1]))
            p2 = (int(history[i][0]),   int(history[i][1]))
            cv2.line(canvas, p1, p2, faded, t, cv2.LINE_AA)

    def _draw_fps(self, canvas, fps, frame_idx):
        if not self.show_fps or fps is None:
            return
        h = canvas.shape[0]
        txt = f"FPS: {fps:.1f}"
        if frame_idx is not None:
            txt += f"  Frame: {frame_idx}"
        cv2.putText(canvas, txt, (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,128), 2, cv2.LINE_AA)

    def draw_gt_overlay(self, canvas, gt_boxes, color=(0,255,255)):
        for box in gt_boxes:
            x1, y1, x2, y2 = box.bbox_xyxy.astype(int)
            _dashed_rect(canvas, (x1,y1), (x2,y2), color)
        return canvas


def _dashed_rect(img, pt1, pt2, color, thickness=1, dash=8):
    x1,y1 = pt1; x2,y2 = pt2
    def dline(p1,p2):
        dx,dy = p2[0]-p1[0], p2[1]-p1[1]
        ln = max(abs(dx),abs(dy))
        if ln==0: return
        steps = max(1, ln//dash)
        for s in range(steps):
            if s%2==0:
                fx=p1[0]+dx*s/steps; fy=p1[1]+dy*s/steps
                tx=p1[0]+dx*(s+1)/steps; ty=p1[1]+dy*(s+1)/steps
                cv2.line(img,(int(fx),int(fy)),(int(tx),int(ty)),color,thickness,cv2.LINE_AA)
    dline((x1,y1),(x2,y1)); dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2)); dline((x1,y2),(x1,y1))