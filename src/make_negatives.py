"""Mine background patches so the model learns what litter-free texture looks like.

Why this is not "hard-negative mining"
-------------------------------------
The obvious move is to take the detections that overlap no ground-truth box and
label them as background. On this dataset that would be actively harmful:
`src/label_noise.py` shows that of the 24 highest-confidence unmatched detections,
21 are visibly real litter that TACO never annotated. Training on those as
negatives would teach the model to suppress real objects.

So this script mines only patches that are background by two independent tests:

1. no ground-truth box overlaps the patch, and
2. the current model does not fire inside the patch above `--conf`.

That second condition makes these *easy* negatives -- gravel, leaves, tile,
soil, sky, wall. They cannot fix a blackboard false positive on their own, but
they do calibrate the objectness head against texture, which is the same
mechanism that produces it.

The patches are written with empty YOLO label files and a `data_neg.yaml` whose
`train` entry is a txt list (original train images + patches), so nothing is
copied or overwritten.

    python src/make_negatives.py --n 400
"""

from __future__ import annotations

import argparse
import random

import cv2
import numpy as np
import torch

from data_utils import ROOT, YOLO_ROOT, load_gt
from models import load_yolo, predict_yolo

NEG_DIR = ROOT / "data" / "negatives"
YAML = ROOT / "data" / "yolo_neg.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    (NEG_DIR / "images").mkdir(parents=True, exist_ok=True)
    (NEG_DIR / "labels").mkdir(parents=True, exist_ok=True)

    records = load_gt("train")
    model = load_yolo(ROOT / "runs" / "yolo" / "best.pt")

    written = 0
    tried = 0
    for rec in records:
        if written >= args.n:
            break
        img = cv2.imread(str(rec["path"]))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = rec["boxes"].numpy()

        pred = predict_yolo(model, [str(rec["path"])], conf=args.conf)[0]
        pb = pred["boxes"].numpy()

        s = args.size
        if h <= s or w <= s:
            continue
        for _ in range(6):
            tried += 1
            x0 = random.randint(0, w - s)
            y0 = random.randint(0, h - s)
            patch_box = np.array([x0, y0, x0 + s, y0 + s], dtype=np.float32)

            # condition 1: no ground truth inside
            if len(gt):
                ix1 = np.maximum(gt[:, 0], patch_box[0])
                iy1 = np.maximum(gt[:, 1], patch_box[1])
                ix2 = np.minimum(gt[:, 2], patch_box[2])
                iy2 = np.minimum(gt[:, 3], patch_box[3])
                inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
                if inter.max() > 0:
                    continue

            # condition 2: the model is quiet here
            if len(pb):
                cx = (pb[:, 0] + pb[:, 2]) / 2
                cy = (pb[:, 1] + pb[:, 3]) / 2
                inside = (cx >= x0) & (cx <= x0 + s) & (cy >= y0) & (cy <= y0 + s)
                if inside.any():
                    continue

            name = f"neg_{written:04d}"
            cv2.imwrite(str(NEG_DIR / "images" / f"{name}.jpg"), img[y0:y0 + s, x0:x0 + s])
            (NEG_DIR / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            written += 1
            break

    # train list = original train images + the patches
    train_imgs = sorted((YOLO_ROOT / "images" / "train").glob("*.jpg"))
    neg_imgs = sorted((NEG_DIR / "images").glob("*.jpg"))
    list_path = ROOT / "data" / "train_with_negatives.txt"
    list_path.write_text(
        "\n".join(str(p) for p in train_imgs + neg_imgs), encoding="utf-8"
    )

    import yaml

    base = yaml.safe_load((YOLO_ROOT / "data.yaml").read_text(encoding="utf-8"))
    # keep `path` pointing at the YOLO root so val/test still resolve, and give
    # `train` an absolute file list (ultralytics accepts a .txt of image paths)
    base["path"] = str(YOLO_ROOT)
    base["train"] = str(list_path.resolve())
    YAML.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    print(f"wrote {written} background patches from {tried} candidate crops")
    print(f"  images  {NEG_DIR / 'images'}")
    print(f"  list    {list_path}  ({len(train_imgs)} train + {len(neg_imgs)} patches)")
    print(f"  yaml    {YAML}")


if __name__ == "__main__":
    main()
