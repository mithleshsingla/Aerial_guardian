#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Converts VisDrone MOT annotations → YOLO format for fine-tuning (nc=1).

ANNOTATION HANDLING:
  score=1, category∈{1,2} → YOLO class 0 "person"  (written to label file)
  score=0                  → ignore region           (SKIPPED entirely)
  Person boxes that fall >50% inside a score=0 region → also SKIPPED

WHY SKIP PERSONS INSIDE IGNORE REGIONS:
  If a person box overlaps an ignore region by >50%, any detector prediction
  on that person gets penalised as a false positive during YOLO loss computation
  (because the GT box is there, but the model may fire slightly off-centre).
  Removing these boxes prevents unfair FP penalty and reduces the
  person→background confusion matrix cell.

SUBSAMPLING:
  VisDrone-MOT-train has ~56k frames across 56 sequences.
  --subsample 3 → ~19k frames. Enough for strong fine-tuning, 3× faster.
  Val set is never subsampled.
"""

import argparse
import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2

PERSON_CATEGORIES = {1, 2}
YOLO_PERSON_ID    = 0


# ── Geometry helpers ──────────────────────────────────────────────────────────

def box_inside(inner, outer) -> bool:
    """
    Returns True if >50% of the inner box's area is inside the outer box.
    Used to filter person boxes that sit inside annotator-marked ignore regions.
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    inter_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inter   = inter_w * inter_h
    inner_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    return (inter / inner_area) > 0.5


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Pixel xyxy → normalised YOLO [cx, cy, w, h], clamped to [0,1]."""
    cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
    cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
    bw = max(0.0, min(1.0, (x2 - x1) / img_w))
    bh = max(0.0, min(1.0, (y2 - y1) / img_h))
    return cx, cy, bw, bh


# ── Annotation parser ─────────────────────────────────────────────────────────

def parse_annotation_file(ann_path: str):
    """
    Parse a VisDrone MOT .txt file.

    Returns:
        person_boxes: {frame_id: [(x1,y1,x2,y2), ...]}  active persons
        ignore_boxes: {frame_id: [(x1,y1,x2,y2), ...]}  score=0 regions
    """
    person_boxes: dict = {}
    ignore_boxes: dict = {}

    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            left     = float(parts[2])
            top      = float(parts[3])
            w        = float(parts[4])
            h        = float(parts[5])
            score    = int(parts[6])
            category = int(parts[7])

            if w <= 0 or h <= 0:
                continue

            box = (left, top, left + w, top + h)

            if score == 0:
                ignore_boxes.setdefault(frame_id, []).append(box)
            elif category in PERSON_CATEGORIES:
                person_boxes.setdefault(frame_id, []).append(box)

    return person_boxes, ignore_boxes


def filter_persons(person_boxes, ignore_boxes):
    """
    Remove person boxes that fall >50% inside any ignore region.
    Returns filtered list of person boxes.
    """
    if not ignore_boxes:
        return person_boxes
    filtered = []
    for pbox in person_boxes:
        inside_ignore = any(box_inside(pbox, ibox) for ibox in ignore_boxes)
        if not inside_ignore:
            filtered.append(pbox)
    return filtered


# ── Dataset converter ─────────────────────────────────────────────────────────

def process_split(
    dataset_root: str,
    out_dir: str,
    split: str,
    skip_empty: bool,
    subsample: int,
):
    seq_base = os.path.join(dataset_root, "sequences")
    ann_base = os.path.join(dataset_root, "annotations")

    img_out = os.path.join(out_dir, "images", split)
    lbl_out = os.path.join(out_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    sequences = sorted([
        d for d in os.listdir(seq_base)
        if os.path.isdir(os.path.join(seq_base, d))
    ])

    total_imgs     = 0
    total_boxes    = 0
    filtered_count = 0
    skipped_empty  = 0

    for seq in tqdm(sequences, desc=f"  {split}"):
        ann_file = os.path.join(ann_base, f"{seq}.txt")
        seq_dir  = os.path.join(seq_base, seq)

        if not os.path.exists(ann_file):
            continue

        all_person_boxes, all_ignore_boxes = parse_annotation_file(ann_file)
        frame_files = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))

        for idx, fpath in enumerate(frame_files):
            if idx % subsample != 0:
                continue

            frame_id = int(Path(fpath).stem)
            raw_pboxes = all_person_boxes.get(frame_id, [])
            iboxes     = all_ignore_boxes.get(frame_id, [])

            # Filter persons inside ignore regions
            pboxes = filter_persons(raw_pboxes, iboxes)
            filtered_count += len(raw_pboxes) - len(pboxes)

            if skip_empty and len(pboxes) == 0:
                skipped_empty += 1
                continue

            img = cv2.imread(fpath)
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            stem    = f"{seq}_{frame_id:07d}"
            dst_img = os.path.join(img_out, f"{stem}.jpg")
            dst_lbl = os.path.join(lbl_out, f"{stem}.txt")

            shutil.copy2(fpath, dst_img)

            with open(dst_lbl, "w") as lf:
                for (x1, y1, x2, y2) in pboxes:
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
                    lf.write(f"{YOLO_PERSON_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    total_boxes += 1

            total_imgs += 1

    print(
        f"  {split}: {total_imgs} images | "
        f"{total_boxes} person boxes | "
        f"{filtered_count} boxes removed (inside ignore regions) | "
        f"skipped {skipped_empty} empty frames"
    )
    return total_imgs, total_boxes


def write_yaml(out_dir: str) -> str:
    yaml_path = os.path.join(out_dir, "visdrone_person.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""# VisDrone person detection — nc=1
# Auto-generated by prepare_dataset.py
# Persons inside score=0 ignore regions have been removed from labels.

path: {os.path.abspath(out_dir)}
train: images/train
val:   images/val

nc: 1
names: ['person']
""")
    print(f"  YAML → {yaml_path}")
    return yaml_path


def main():
    p = argparse.ArgumentParser(
        description="Convert VisDrone MOT → YOLO format (nc=1, ignore-filtered)"
    )
    p.add_argument("--train",      required=True,  help="VisDrone2019-MOT-train/")
    p.add_argument("--val",        required=True,  help="VisDrone2019-MOT-val/")
    p.add_argument("--out",        required=True,  help="Output YOLO dataset directory")
    p.add_argument("--skip-empty", action="store_true",
                   help="Skip frames with no valid person annotations after filtering")
    p.add_argument("--subsample",  type=int, default=3,
                   help="Keep 1-in-N train frames (default=3)")
    args = p.parse_args()

    print(f"\n[Prepare] Output    → {args.out}")
    print(f"[Prepare] Subsample → every {args.subsample} train frames")
    print(f"[Prepare] Filter    → persons inside ignore regions removed\n")

    os.makedirs(args.out, exist_ok=True)

    process_split(args.train, args.out, "train",
                  skip_empty=args.skip_empty, subsample=args.subsample)
    process_split(args.val,   args.out, "val",
                  skip_empty=False, subsample=1)

    yaml_path = write_yaml(args.out)
    print(f"\n[Prepare] Done! Next:")
    print(f"  python scripts/fine_tune.py --data {yaml_path}")


if __name__ == "__main__":
    main()