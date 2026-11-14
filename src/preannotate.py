"""Seed the campus COCO file with model predictions so a human only has to correct.

Annotating 120 frames from scratch takes hours; correcting pre-filled boxes takes
minutes. This writes the trained YOLO's boxes at a deliberately low confidence
(`--conf 0.15`) into `data/campus/_annotations.coco.json`.

Important, and stated in the README: boxes produced this way are a *draft*.
The campus set must be corrected by a human before it is used as ground truth,
and the correction must not be skipped. A test set whose labels came from the
model being tested is not a test set -- it would flatter YOLO and, worse, the
evaluation would be circular.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_utils import CLASSES, ROOT
from models import load_yolo, predict_yolo

CAMPUS = ROOT / "data" / "campus"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15)
    args = ap.parse_args()

    p = CAMPUS / "_annotations.coco.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- run extract_frames.py first")
    data = json.loads(p.read_text(encoding="utf-8"))
    if data["annotations"]:
        raise SystemExit(
            "this file already has annotations; refusing to overwrite hand-corrected "
            "labels. Delete them first if you really want to re-seed."
        )

    model = load_yolo(ROOT / "runs" / "yolo" / "best.pt")
    paths = [CAMPUS / "images" / im["file_name"] for im in data["images"]]
    preds = predict_yolo(model, [str(x) for x in paths], conf=args.conf)

    ann_id = 1
    n_boxes = 0
    for im, pred in zip(data["images"], preds):
        for (x1, y1, x2, y2), sc, lab in zip(
            pred["boxes"].numpy(), pred["scores"].numpy(), pred["labels"].numpy()
        ):
            x1 = max(0.0, min(float(x1), im["width"] - 1))
            y1 = max(0.0, min(float(y1), im["height"] - 1))
            x2 = max(0.0, min(float(x2), im["width"] - 1))
            y2 = max(0.0, min(float(y2), im["height"] - 1))
            w, h = x2 - x1, y2 - y1
            if w < 2 or h < 2:
                continue
            data["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": im["id"],
                    "category_id": int(lab),
                    "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                    "area": round(w * h, 1),
                    "iscrowd": 0,
                    "score": round(float(sc), 3),
                    "reviewed": False,
                }
            )
            ann_id += 1
            n_boxes += 1

    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"seeded {n_boxes} boxes across {len(data['images'])} frames at conf {args.conf}")
    print("NEXT: correct these boxes by hand (add misses, delete false positives,")
    print("      fix classes), then set reviewed=true on frames you have checked.")


if __name__ == "__main__":
    main()
