"""Turn a recorded campus clip into an annotatable frame set.

Pipeline: video -> sample at a fixed rate -> drop near-duplicate frames -> JPEGs
plus a COCO skeleton with the right image entries and an empty annotation list.

Why de-duplicate
----------------
Walking at a normal pace with a 30 fps camera, consecutive frames are nearly
identical; annotating both wastes hours and inflates the test set with the same
object counted twice. Frames are compared against the last *kept* frame using
mean absolute difference on a 64x64 grayscale thumbnail, and a frame is kept only
when it differs by more than `--min-diff`.

The COCO skeleton is what `preannotate.py` fills in, and what a human then
corrects in any COCO-capable tool (CVAT, Label Studio, or the built-in editor in
`annotate_helper.py`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from data_utils import CLASSES, ROOT

OUT_DEFAULT = ROOT / "data" / "campus"


def thumb(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to the recorded clip")
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate")
    ap.add_argument("--min-diff", type=float, default=6.0, help="mean abs diff to keep a frame")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / args.fps)))
    print(f"{args.video}: {src_fps:.1f} fps, taking every {step}th frame")

    out_dir = Path(args.out) / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    prev = None
    n_read = n_kept = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            n_read += 1
            t = thumb(frame)
            diff = float(np.abs(t - prev).mean()) if prev is not None else 999.0
            if diff >= args.min_diff:
                name = f"campus_{n_kept:05d}.jpg"
                cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                h, w = frame.shape[:2]
                images.append({"id": n_kept, "file_name": name, "width": w, "height": h})
                n_kept += 1
                prev = t
                if args.max_frames and n_kept >= args.max_frames:
                    break
        idx += 1
    cap.release()

    coco = {
        "info": {
            "description": "Campus litter frames (out-of-domain test set)",
            "source_video": str(args.video),
            "sampled_fps": args.fps,
            "min_diff": args.min_diff,
        },
        "licenses": [],
        "images": images,
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(CLASSES)],
    }
    p = Path(args.out) / "_annotations.coco.json"
    p.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"sampled {n_read} frames, kept {n_kept} -> {out_dir}")
    print(f"wrote {p} (annotations empty: run preannotate.py, then correct by hand)")


if __name__ == "__main__":
    main()
