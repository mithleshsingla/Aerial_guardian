"""
annotation_loader.py
====================
Loads VisDrone MOT ground-truth annotation files.

Format (CSV, no header):
  frame_id, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
  score, object_category, truncation, occlusion

VisDrone category IDs:
  0  = ignored region  (skip)
  1  = pedestrian      ← our main target
  2  = people (group)  ← also a person target
  3  = bicycle
  4  = car
  5  = van
  6  = truck
  7  = tricycle
  8  = awning-tricycle
  9  = bus
  10 = motor
  11 = others

Truncation flags: 0=no truncation, 1=truncated
Occlusion flags:  0=no occlusion, 1=partial, 2=heavy

Usage:
    loader = AnnotationLoader("annotations/uav0000086_00000_v.txt")
    frame_gt = loader.get_frame(102)   # list of GTBox for frame 102
    all_ids  = loader.track_ids        # set of all target_ids in file
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import numpy as np


# VisDrone category names (index = category id)
VISDRONE_CATEGORIES = {
    0:  "ignored",
    1:  "pedestrian",
    2:  "people",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tricycle",
    9:  "bus",
    10: "motor",
    11: "others",
}

# Categories we treat as "person"
PERSON_CATEGORY_IDS = {1, 2}   # pedestrian + people


@dataclass
class GTBox:
    """One ground-truth bounding box."""
    frame_id:    int
    target_id:   int
    bbox_xyxy:   np.ndarray   # [x1, y1, x2, y2]
    score:       int           # 0=ignore, 1=active
    category_id: int
    category_name: str
    truncation:  int
    occlusion:   int

    @property
    def is_person(self) -> bool:
        return self.category_id in PERSON_CATEGORY_IDS

    @property
    def is_active(self) -> bool:
        """score==0 means annotator marked it as 'ignore' for evaluation."""
        return self.score == 1


class AnnotationLoader:
    """
    Parses a single VisDrone MOT annotation .txt file and provides
    per-frame access to ground-truth boxes.
    """

    def __init__(self, ann_path: str, person_only: bool = True):
        """
        Args:
            ann_path:    Path to annotation .txt file
            person_only: If True, only load pedestrian + people categories
        """
        self.ann_path = Path(ann_path)
        self.person_only = person_only

        # frame_id → list of GTBox
        self._data: Dict[int, List[GTBox]] = {}
        self._track_ids: Set[int] = set()
        self._frame_ids: Set[int] = set()

        self._load()

    def _load(self):
        """Parse annotation file into internal dict."""
        if not self.ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.ann_path}")

        with open(self.ann_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue  # malformed line

                frame_id    = int(parts[0])
                target_id   = int(parts[1])
                left        = float(parts[2])
                top         = float(parts[3])
                width       = float(parts[4])
                height      = float(parts[5])
                score       = int(parts[6])
                category_id = int(parts[7])
                truncation  = int(parts[8]) if len(parts) > 8 else 0
                occlusion   = int(parts[9]) if len(parts) > 9 else 0

                # Skip ignored regions
                if category_id == 0:
                    continue

                # Optionally filter to person-only
                if self.person_only and category_id not in PERSON_CATEGORY_IDS:
                    continue

                # Convert [left, top, w, h] → [x1, y1, x2, y2]
                bbox_xyxy = np.array([
                    left,
                    top,
                    left + width,
                    top + height,
                ], dtype=np.float32)

                box = GTBox(
                    frame_id=frame_id,
                    target_id=target_id,
                    bbox_xyxy=bbox_xyxy,
                    score=score,
                    category_id=category_id,
                    category_name=VISDRONE_CATEGORIES.get(category_id, "unknown"),
                    truncation=truncation,
                    occlusion=occlusion,
                )

                if frame_id not in self._data:
                    self._data[frame_id] = []
                self._data[frame_id].append(box)
                self._track_ids.add(target_id)
                self._frame_ids.add(frame_id)

    def get_frame(self, frame_id: int) -> List[GTBox]:
        """Return all GT boxes for a given frame (empty list if none)."""
        return self._data.get(frame_id, [])

    def get_active_persons(self, frame_id: int) -> List[GTBox]:
        """Return only active (score=1) person boxes for a frame."""
        return [b for b in self.get_frame(frame_id)
                if b.is_person and b.is_active]

    @property
    def track_ids(self) -> Set[int]:
        return self._track_ids

    @property
    def frame_ids(self) -> Set[int]:
        return self._frame_ids

    @property
    def num_frames(self) -> int:
        return len(self._frame_ids)

    def summary(self) -> str:
        total_boxes = sum(len(v) for v in self._data.values())
        person_boxes = sum(
            len(self.get_active_persons(f)) for f in self._frame_ids
        )
        return (
            f"Annotation: {self.ann_path.name}\n"
            f"  Frames with annotations : {self.num_frames}\n"
            f"  Unique track IDs        : {len(self._track_ids)}\n"
            f"  Total boxes             : {total_boxes}\n"
            f"  Active person boxes     : {person_boxes}\n"
        )