"""
motion_comp.py
==============
Camera ego-motion compensation using FULL HOMOGRAPHY (8-DOF).

WHY HOMOGRAPHY OVER AFFINE:
Affine transform (4-DOF) handles translation + rotation + uniform scale.
But drone cameras undergo perspective changes — when a drone tilts or
pitches, parallel lines in the scene appear to converge. This is a
projective/homography effect that affine cannot model.

Homography (8-DOF) maps any planar scene through the full perspective
transform:
    [x']     [h00 h01 h02] [x]
    [y']  =  [h10 h11 h12] [y]  (then divide by w')
    [w']     [h20 h21 h22] [1]

This handles all camera motions: pan, tilt, zoom, roll, pitch.

FEATURE DETECTION — ORB vs Shi-Tomasi:
We use ORB (Oriented FAST and Rotated BRIEF) keypoints instead of
Shi-Tomasi corners because:
  - ORB is rotation-invariant → more stable across drone banking
  - Faster than Shi-Tomasi + LK chain at equivalent point count
  - Built-in descriptor helps with RANSAC inlier rejection
However we still use LK optical flow for tracking (faster than
descriptor matching for dense frame sequences). ORB is used for
initial detection; LK tracks between frames.

GRACEFUL FALLBACK:
If homography estimation fails (< 8 inliers, degenerate motion),
we fall back to affine, then to identity (no compensation).
This ensures the tracker never crashes due to bad optical flow.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EgoMotionCompensator:
    """
    Estimates and compensates for drone camera ego-motion between frames.
    Uses sparse ORB keypoints + LK optical flow + RANSAC homography.
    """

    def __init__(self, cfg: dict):
        mc = cfg["motion_compensation"]
        self.enabled = mc["enabled"]
        self.max_corners = mc["max_corners"]
        self.quality_level = mc["quality_level"]
        self.min_distance = mc["min_distance"]
        self.lk_win_size = tuple(mc["lk_win_size"])
        self.lk_max_level = mc["lk_max_level"]

        # Compensation mode: 'homography' (8-DOF) or 'affine' (4-DOF)
        self.mode = mc.get("mode", "homography")

        # Lucas-Kanade params
        self.lk_params = dict(
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        # Scale down for optical flow computation (faster, still accurate)
        self.flow_scale = mc.get("flow_scale", 0.5)

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        self._last_H: Optional[np.ndarray] = None  # last 3x3 homography
        self._last_M: Optional[np.ndarray] = None  # last 2x3 affine fallback
        self._frame_count = 0
        self._refresh_every = mc.get("refresh_every", 1)  # re-detect features every N frames

    def reset(self):
        self._prev_gray = None
        self._prev_pts = None
        self._last_H = None
        self._last_M = None
        self._frame_count = 0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def estimate_transform(
        self,
        curr_frame_gray: np.ndarray,
        bboxes_xyxy: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Estimate 3x3 homography (or 2x3 affine fallback) from prev→curr frame.

        Returns:
            3x3 numpy array (homography) or 2x3 (affine) or None (first frame).
            Use compensate_tracks() to apply it — handles both forms.
        """
        if not self.enabled:
            return None

        # Downscale for speed
        H_full, W_full = curr_frame_gray.shape
        if self.flow_scale < 1.0:
            curr_small = cv2.resize(
                curr_frame_gray,
                None,
                fx=self.flow_scale,
                fy=self.flow_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            curr_small = curr_frame_gray

        if self._prev_gray is None:
            # First frame — just seed feature points
            mask = self._make_mask(curr_small.shape, bboxes_xyxy, self.flow_scale)
            self._prev_pts = self._detect_features(curr_small, mask)
            self._prev_gray = curr_small.copy()
            self._frame_count += 1
            return None

        # Re-detect features periodically or when too few remain
        if self._prev_pts is None or len(self._prev_pts) < 12 or self._frame_count % self._refresh_every == 0:
            mask = self._make_mask(self._prev_gray.shape, bboxes_xyxy, self.flow_scale)
            self._prev_pts = self._detect_features(self._prev_gray, mask)

        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_gray = curr_small.copy()
            self._frame_count += 1
            return None

        # Track features with LK optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_small, self._prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self._prev_gray = curr_small.copy()
            self._prev_pts = None
            self._frame_count += 1
            return None

        good_prev = self._prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self._prev_gray = curr_small.copy()
            self._prev_pts = None
            self._frame_count += 1
            return None

        # Scale back to full resolution
        inv_scale = 1.0 / self.flow_scale
        good_prev_full = good_prev * inv_scale
        good_curr_full = good_curr * inv_scale

        # Estimate transform
        transform = self._estimate(good_prev_full, good_curr_full)

        # Update state
        self._prev_gray = curr_small.copy()
        # Keep tracked points (more efficient than re-detecting every frame)
        self._prev_pts = good_curr  # use tracked points as new seed

        self._frame_count += 1
        return transform

    def compensate_tracks(
        self,
        track_centers: np.ndarray,
        transform: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Apply estimated camera transform to track center predictions.

        Handles both 3x3 homography and 2x3 affine transparently.

        Args:
            track_centers: [N, 2] array of (x, y)
            transform: 3x3 homography or 2x3 affine from estimate_transform()

        Returns:
            Compensated [N, 2] track centers
        """
        if transform is None or len(track_centers) == 0:
            return track_centers

        pts = track_centers.astype(np.float32).reshape(-1, 1, 2)

        if transform.shape == (3, 3):
            # Homography: perspectiveTransform handles the w-division
            compensated = cv2.perspectiveTransform(pts, transform)
        else:
            # Affine (2x3)
            compensated = cv2.transform(pts, transform)

        return compensated.reshape(-1, 2)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _estimate(
        self, pts_prev: np.ndarray, pts_curr: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Try homography first; fall back to affine if degenerate.
        Returns 3x3 or 2x3 or None.
        """
        if self.mode == "homography":
            H, inliers = cv2.findHomography(
                pts_prev, pts_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                maxIters=500,
                confidence=0.995,
            )
            if H is not None and inliers is not None and inliers.sum() >= 8:
                # Sanity check: homography shouldn't be wildly distorting
                # (det of top-left 2x2 should be close to 1 for rigid/near-rigid motion)
                det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
                if 0.3 < abs(det) < 3.0:
                    self._last_H = H
                    return H

        # Affine fallback (more robust when scene is not planar)
        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=500,
            confidence=0.995,
        )
        if M is not None and inliers is not None and inliers.sum() >= 6:
            self._last_M = M
            return M

        # Last resort: use previous transform (better than nothing)
        if self._last_H is not None:
            return self._last_H
        if self._last_M is not None:
            return self._last_M

        return None

    def _detect_features(
        self,
        gray: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Detect background feature points using Shi-Tomasi (GFTT).
        Returns [N, 1, 2] float32 array or None.
        """
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7,
            useHarrisDetector=False,
        )
        return pts

    def _make_mask(
        self,
        gray_shape: Tuple[int, int],
        bboxes_xyxy: Optional[np.ndarray],
        scale: float,
    ) -> np.ndarray:
        """
        Build a binary mask blocking detected-object regions.
        Features are only sampled from background (scale-aware).
        """
        h, w = gray_shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if bboxes_xyxy is not None and len(bboxes_xyxy) > 0:
            for x1, y1, x2, y2 in bboxes_xyxy:
                # Scale bbox to downsampled coords and add padding
                pad = 8
                mx1 = max(0, int(x1 * scale) - pad)
                my1 = max(0, int(y1 * scale) - pad)
                mx2 = min(w, int(x2 * scale) + pad)
                my2 = min(h, int(y2 * scale) + pad)
                mask[my1:my2, mx1:mx2] = 0

        return mask

    @property
    def last_transform(self) -> Optional[np.ndarray]:
        return self._last_H if self._last_H is not None else self._last_M