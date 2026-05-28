"""
tracker.py
==========
ByteTrack MOT + homography ego-motion compensation.

ByteTrack dual-stage association recap:
  Pass 1: High-conf detections (>threshold) ↔ active tracks  (IoU)
  Pass 2: Low-conf detections  (0.1–thresh) ↔ unmatched tracks (IoU)
          → rescues persons whose score dipped due to occlusion/altitude
  Lost tracks kept alive for `lost_track_buffer` frames via Kalman extrapolation
  Re-identified by position proximity if they reappear within that window

Homography compensation:
  Before ByteTrack's IoU association step we apply the estimated
  camera-motion homography to each track's Kalman-predicted centre.
  This "cancels" drone movement from the tracker's perspective so
  association is in scene-space, not image-space.
  Falls back gracefully (affine → identity) if homography is degenerate.
"""

import cv2
import numpy as np
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")
import supervision as sv

from motion_comp import EgoMotionCompensator


class AerialTracker:
    """ByteTrack + homography ego-motion compensation, re-initialisable per sequence."""

    def __init__(self, cfg: dict):
        trk = cfg["tracking"]
        vis = cfg["visualization"]

        self.tail_length = vis["tail_length"]

        self.tracker = sv.ByteTrack(
            track_activation_threshold=trk["track_activation_threshold"],
            lost_track_buffer=trk["lost_track_buffer"],
            minimum_matching_threshold=trk["minimum_matching_threshold"],
            frame_rate=trk["frame_rate"],
        )

        self.compensator = EgoMotionCompensator(cfg)

        # {track_id: [(cx, cy), ...]}
        self.track_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self._frame_idx = 0

    def reset(self):
        """Call at the start of each new sequence."""
        self.tracker.reset()
        self.compensator.reset()
        self.track_history.clear()
        self._frame_idx = 0

    def update(
        self,
        detections: "sv.Detections",
        frame_bgr: np.ndarray,
    ) -> "sv.Detections":
        """One frame: compensate → associate → update histories."""
        gray      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bboxes    = detections.xyxy if len(detections) > 0 else None
        transform = self.compensator.estimate_transform(gray, bboxes)

        if transform is not None:
            self._apply_compensation(transform)

        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is not None and len(tracked) > 0:
            for i, tid in enumerate(tracked.tracker_id):
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                hist = self.track_history[int(tid)]
                hist.append((cx, cy))
                if len(hist) > self.tail_length:
                    hist.pop(0)

        self._frame_idx += 1
        return tracked

    def get_tail(self, track_id: int) -> List[Tuple[float, float]]:
        return self.track_history.get(int(track_id), [])

    # ------------------------------------------------------------------ #
    # Kalman state patching (camera-motion compensation)
    # ------------------------------------------------------------------ #

    def _apply_compensation(self, transform: np.ndarray):
        """
        Shift ByteTrack's Kalman-predicted track centres by the camera
        motion transform before IoU-based association.

        ByteTrack Kalman state: [cx, cy, aspect, height, v_cx, v_cy, v_ar, v_h]
        We update indices 0 (cx) and 1 (cy).

        supervision's internal attribute names have changed across versions,
        so we try multiple access patterns and fail silently.
        """
        try:
            all_tracks = []
            for attr in ("tracked_stracks", "_tracked_stracks",
                         "lost_stracks",    "_lost_stracks"):
                if hasattr(self.tracker, attr):
                    all_tracks.extend(list(getattr(self.tracker, attr)))
            # supervision ≥ 0.25 style
            if not all_tracks and hasattr(self.tracker, "_tracks"):
                all_tracks = list(self.tracker._tracks.values())

            is_homography = (transform.shape == (3, 3))

            for track in all_tracks:
                if not (hasattr(track, "mean") and track.mean is not None):
                    continue
                cx = float(track.mean[0])
                cy = float(track.mean[1])
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                if is_homography:
                    new_pt = cv2.perspectiveTransform(pt, transform)
                else:
                    new_pt = cv2.transform(pt, transform)
                track.mean[0] = float(new_pt[0, 0, 0])
                track.mean[1] = float(new_pt[0, 0, 1])

        except Exception:
            pass   # graceful: skip compensation this frame