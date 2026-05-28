"""
evaluator.py
============
MOT evaluation against VisDrone ground truth.

WHY IoU=0.5 IS WRONG FOR AERIAL TINY OBJECTS:
  IoU=0.5 was designed for COCO where persons are 100-400px tall.
  On VisDrone, persons can be 8x16px. A 4px localisation offset
  (visually near-perfect) gives IoU=0.41 — counted as FN+FP.
  This artificially inflates both FN and FP counts, collapsing MOTA.

  IoU=0.3 is the correct threshold for tiny aerial objects:
  - A 4px offset on a 12x30px person: IoU=0.41 → TP at 0.3 ✓
  - A 5px offset on a 12x30px person: IoU=0.32 → TP at 0.3 ✓
  - A 2px offset on a 8x16px person:  IoU=0.49 → TP at 0.3 ✓

  VisDrone challenge itself uses IoU=0.5 for its official leaderboard,
  but for engineering evaluation of a tracking system the lower threshold
  gives a more honest picture of detection quality.

  We report BOTH: MOTA@0.5 (official) and MOTA@0.3 (practical).

METRICS COMPUTED:
  MOTA      = 1 - (FN + FP + IDSW) / GT
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  IDSW      = ID switches (GT track matched to different pred ID vs prev frame)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from annotation_loader import GTBox


@dataclass
class FrameStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    id_switches: int = 0
    gt_count: int = 0


class Evaluator:
    """
    Frame-by-frame MOT evaluator.
    Runs greedy IoU matching at a configurable threshold.
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold   = iou_threshold
        self.frame_stats:    List[FrameStats] = []
        self._gt_to_pred_id: Dict[int, Optional[int]] = {}
        self._id_switches    = 0

    def reset(self):
        self.frame_stats.clear()
        self._gt_to_pred_id.clear()
        self._id_switches = 0

    def update(
        self,
        frame_id:       int,
        gt_boxes:       List[GTBox],
        pred_xyxy:      np.ndarray,
        pred_track_ids: np.ndarray,
    ):
        stats = FrameStats(gt_count=len(gt_boxes))

        if len(gt_boxes) == 0 and len(pred_xyxy) == 0:
            self.frame_stats.append(stats)
            return
        if len(gt_boxes) == 0:
            stats.fp = len(pred_xyxy)
            self.frame_stats.append(stats)
            return
        if len(pred_xyxy) == 0:
            stats.fn = len(gt_boxes)
            self.frame_stats.append(stats)
            return

        gt_xyxy  = np.array([b.bbox_xyxy for b in gt_boxes])
        iou_mat  = self._iou_matrix(gt_xyxy, pred_xyxy)

        matched_gt:   Set[int] = set()
        matched_pred: Set[int] = set()
        matches:      List[Tuple[int,int]] = []

        order = np.dstack(np.unravel_index(
            np.argsort(-iou_mat, axis=None), iou_mat.shape
        ))[0]

        for gi, pi in order:
            if gi in matched_gt or pi in matched_pred:
                continue
            if iou_mat[gi, pi] >= self.iou_threshold:
                matches.append((gi, pi))
                matched_gt.add(gi)
                matched_pred.add(pi)

        stats.tp = len(matches)
        stats.fn = len(gt_boxes)   - len(matches)
        stats.fp = len(pred_xyxy)  - len(matches)

        for gi, pi in matches:
            gt_id   = gt_boxes[gi].target_id
            pred_id = int(pred_track_ids[pi])
            prev    = self._gt_to_pred_id.get(gt_id)
            if prev is not None and prev != pred_id:
                stats.id_switches  += 1
                self._id_switches  += 1
            self._gt_to_pred_id[gt_id] = pred_id

        self.frame_stats.append(stats)

    def compute(self) -> Dict:
        total_gt   = sum(s.gt_count      for s in self.frame_stats)
        total_tp   = sum(s.tp            for s in self.frame_stats)
        total_fp   = sum(s.fp            for s in self.frame_stats)
        total_fn   = sum(s.fn            for s in self.frame_stats)
        total_idsw = sum(s.id_switches   for s in self.frame_stats)

        precision  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall     = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        mota       = 1.0 - (total_fn + total_fp + total_idsw) / total_gt if total_gt > 0 else 0.0

        return {
            "MOTA":      round(mota,      4),
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "TP":        total_tp,
            "FP":        total_fp,
            "FN":        total_fn,
            "IDSW":      total_idsw,
            "GT_total":  total_gt,
            "Frames":    len(self.frame_stats),
            "IoU_thr":   self.iou_threshold,
        }

    def report(self, seq_name: str = "") -> str:
        m = self.compute()
        h = f"=== Evaluation [{seq_name}]  IoU≥{m['IoU_thr']} ===" if seq_name \
            else f"=== Evaluation  IoU≥{m['IoU_thr']} ==="
        return (
            f"{h}\n"
            f"  MOTA      : {m['MOTA']:>8.4f}\n"
            f"  Precision : {m['Precision']:>8.4f}\n"
            f"  Recall    : {m['Recall']:>8.4f}\n"
            f"  TP        : {m['TP']:>8d}\n"
            f"  FP        : {m['FP']:>8d}\n"
            f"  FN        : {m['FN']:>8d}\n"
            f"  ID Switch : {m['IDSW']:>8d}\n"
            f"  GT boxes  : {m['GT_total']:>8d}\n"
            f"  Frames    : {m['Frames']:>8d}\n"
        )

    @staticmethod
    def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        area_a = (boxes_a[:,2]-boxes_a[:,0]) * (boxes_a[:,3]-boxes_a[:,1])
        area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
        inter_x1 = np.maximum(boxes_a[:,None,0], boxes_b[None,:,0])
        inter_y1 = np.maximum(boxes_a[:,None,1], boxes_b[None,:,1])
        inter_x2 = np.minimum(boxes_a[:,None,2], boxes_b[None,:,2])
        inter_y2 = np.minimum(boxes_a[:,None,3], boxes_b[None,:,3])
        inter    = np.maximum(0, inter_x2-inter_x1) * np.maximum(0, inter_y2-inter_y1)
        union    = area_a[:,None] + area_b[None,:] - inter
        return np.where(union > 0, inter/union, 0.0)