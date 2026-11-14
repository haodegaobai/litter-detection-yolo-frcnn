"""Build a synthetic "walk-past" clip from test images, to exercise the live path.

Purpose: `live_demo.py` needs a video source to be tested on a machine with no
camera, and the fallback demo needs *something* to play if the webcam misbehaves.
This stitches test-set stills into a moving clip with a slow pan and a slight
zoom, which is enough to verify that capture -> preprocess -> inference -> NMS ->
draw -> record all work together.

It is NOT a substitute for real footage: the images come from TACO, so a
domain-shift number computed on this clip would be meaningless (it is in-domain
by construction). Use it to test the plumbing, then record the real clip with
`live_demo.py --record`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from data_utils import ROOT, load_gt

OUT = ROOT / "demo" / "test_clip_synthetic.mp4"


def ken_burns(img: np.ndarray, n: int, w: int, h: int) -> list[np.ndarray]:
    """Return n frames panning slowly across the image, zoomed to fill w x h."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih) * 1.15  # slight overscan so the pan has room
    big = cv2.resize(img, (int(iw * scale), int(ih * scale)))
    bh, bw = big.shape[:2]
    frames = []
    for i in range(n):
        t = i / max(1, n - 1)
        x = int((bw - w) * t)
        y = int((bh - h) * (0.5 + 0.1 * np.sin(t * np.pi)))
        frames.append(big[y:y + h, x:x + w].copy())
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=6)
    ap.add_argument("--seconds-per-image", type=float, default=2.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--pick",
        default=None,
        help="comma-separated test file names to use, in order. Without it the "
             "first N test images with >= 2 boxes are taken, which can land on "
             "indoor product shots that look nothing like litter on the ground.",
    )
    args = ap.parse_args()

    all_recs = load_gt("test")
    if args.pick:
        # accept names with or without the extension
        wanted = [w.strip() for w in args.pick.split(",") if w.strip()]
        by_stem = {Path(r["path"].name).stem: r for r in all_recs}
        missing = [w for w in wanted if Path(w).stem not in by_stem]
        if missing:
            raise SystemExit(f"not in the test split: {missing}")
        recs = [by_stem[Path(w).stem] for w in wanted]
    else:
        recs = [r for r in all_recs if r["boxes"].shape[0] >= 2][: args.images]
    if not recs:
        raise SystemExit("no test images available; run prepare_data.py first")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height)
    )
    n_per = int(args.seconds_per_image * args.fps)

    total = 0
    for r in recs:
        img = cv2.imread(str(r["path"]))
        if img is None:
            continue
        for f in ken_burns(img, n_per, args.width, args.height):
            writer.write(f)
            total += 1
    writer.release()
    print(f"wrote {out} ({total} frames, {total / args.fps:.1f}s, {args.width}x{args.height})")
    print("synthetic: TACO stills, in-domain. For testing the pipeline only.")


if __name__ == "__main__":
    main()
