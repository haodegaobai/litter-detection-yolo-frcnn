"""Shared data utilities for the campus-litter project.

One COCO reader, one class list, used by BOTH detectors. Every number in the
results table is produced from ground truth parsed here, so the comparison
cannot drift between models.

Class taxonomy
--------------
TACO ships 60 fine-grained categories with a very long tail (12 of them have
fewer than 5 instances). Detecting 60 classes on 1500 images is not a learnable
problem, and mAP over that head would be dominated by classes with no signal.
The 60 categories are therefore folded into 6 material superclasses.

Folding rule (fixed before any model was trained, not tuned afterwards):
  1. map every TACO category to its material family;
  2. if a superclass ends up with < 100 instances, merge it into `other`;
  3. if `other` itself ends up with < 100 instances, drop it entirely.

Measured instance counts after folding (all 4784 annotations are kept):
  plastic 2112 | paper_cardboard 498 | metal 552 | glass 254 | cigarette 667 | other 701
No superclass fell below the 100-instance floor, so step 2 never triggered.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
COCO_ROOT = ROOT / "data" / "coco" / "litter"
YOLO_ROOT = ROOT / "data" / "yolo"

CLASSES = ["plastic", "paper_cardboard", "metal", "glass", "cigarette", "other"]
NUM_CLASSES = len(CLASSES)

SPLIT_ALIAS = {"val": "valid", "valid": "valid", "train": "train", "test": "test"}

# TACO category name -> superclass. 60 names, each mapped exactly once.
SUPERCLASS_MAP: dict[str, str] = {}
for _names, _sc in [
    (
        [
            "Other plastic bottle", "Clear plastic bottle", "Plastic bottle cap",
            "Disposable plastic cup", "Other plastic cup", "Plastic lid",
            "Other plastic", "Plastic film", "Six pack rings", "Garbage bag",
            "Other plastic wrapper", "Single-use carrier bag", "Polypropylene bag",
            "Crisp packet", "Spread tub", "Tupperware", "Disposable food container",
            "Other plastic container", "Plastic glooves", "Plastic utensils",
            "Squeezable tube", "Plastic straw",
        ],
        "plastic",
    ),
    (
        [
            "Carded blister pack", "Toilet tube", "Other carton", "Egg carton",
            "Drink carton", "Corrugated carton", "Meal carton", "Pizza box",
            "Paper cup", "Magazine paper", "Tissues", "Wrapping paper",
            "Normal paper", "Paper bag", "Plastified paper bag", "Paper straw",
        ],
        "paper_cardboard",
    ),
    (
        [
            "Aluminium foil", "Battery", "Aluminium blister pack", "Metal bottle cap",
            "Food Can", "Aerosol", "Drink can", "Metal lid", "Pop tab", "Scrap metal",
        ],
        "metal",
    ),
    (["Glass bottle", "Broken glass", "Glass cup", "Glass jar"], "glass"),
    (["Cigarette"], "cigarette"),
    (
        [
            "Foam cup", "Food waste", "Foam food container", "Rope & strings",
            "Shoe", "Styrofoam piece", "Unlabeled litter",
        ],
        "other",
    ),
]:
    for _n in _names:
        SUPERCLASS_MAP[_n] = _sc


def load_gt(split: str) -> list[dict]:
    """Return one record per image: xyxy boxes in pixel coords + 0-based labels.

    Reads the *prepared* COCO files under data/coco/litter/<split>/, i.e. the
    files produced by prepare_data.py after folding and resizing. Both models
    read from here and nowhere else.
    """
    split = SPLIT_ALIAS[split]
    with open(COCO_ROOT / split / "_annotations.coco.json", encoding="utf-8") as fh:
        data = json.load(fh)

    # prepare_data.py writes contiguous 0-based category ids, so the COCO
    # category_id is already the index into CLASSES -- no remapping needed here.
    per_image = defaultdict(list)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        per_image[ann["image_id"]].append((x, y, x + w, y + h, ann["category_id"]))

    records = []
    for img in data["images"]:
        items = per_image.get(img["id"], [])
        if items:
            boxes = torch.tensor([it[:4] for it in items], dtype=torch.float32)
            labels = torch.tensor([it[4] for it in items], dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        records.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "path": COCO_ROOT / split / img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "boxes": boxes,
                "labels": labels,
            }
        )
    return records


def to_torchmetric_input(records: list[dict]) -> list[dict]:
    """Ground truth in the shape torchmetrics expects (no scores for GT)."""
    return [{"boxes": r["boxes"], "labels": r["labels"]} for r in records]
