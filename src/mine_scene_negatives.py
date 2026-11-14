"""Mine negative training frames from real scenes, so the model learns "not litter".

The failure this fixes
----------------------
The detector answers "what material is this", not "is this litter". Given a
cardboard box on a desk it says `paper_cardboard` -- correctly, and uselessly.
Given a face it says `plastic`, because plastic is 44% of the training instances
and nothing in training ever said "this is nothing".

TACO contains no negative images at all: every frame has litter in it. So the
model has never seen a desk, a hand, a drawer or a hedge labelled as background.

The cheapest fix is not more litter data -- it is negative data, and it is
already on disk: the demo recordings and the campus clips are exactly the kind of
scene the model gets wrong. This script turns them into training frames with
empty label files.

    python src/mine_scene_negatives.py --per-video 120

Writes `data/negatives_scene/` and a `data/yolo_scene.yaml` whose train list is
the original train images plus the mined frames.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
from pathlib import Path

import cv2

from data_utils import ROOT, YOLO_ROOT

OUT = ROOT / "data" / "negatives_scene"
LIST = ROOT / "data" / "train_with_scene_negatives.txt"
YAML = ROOT / "data" / "yolo_scene.yaml"


def frames_from_video(path: Path, want: int, tmp: Path) -> list[Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    for old in tmp.glob("*.jpg"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", "fps=1,scale=640:-2",
         str(tmp / "v_%04d.jpg")],
        check=True,
    )
    got = sorted(tmp.glob("*.jpg"))
    if len(got) <= want:
        return got
    step = len(got) / want
    return [got[int(i * step)] for i in range(want)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-video", type=int, default=120)
    ap.add_argument("--campus", action="store_true", default=True)
    args = ap.parse_args()

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "data" / "_tmp_frames"

    written = 0
    sources = sorted(glob.glob(str(ROOT / "demo" / "raw" / "*.mp4")))
    for v in sources:
        for f in frames_from_video(Path(v), args.per_video, tmp):
            im = cv2.imread(str(f))
            if im is None:
                continue
            name = f"scene_{written:04d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), im)
            (OUT / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            written += 1
        print(f"  {Path(v).name}: total {written}")

    if args.campus:
        for f in sorted(glob.glob(str(ROOT / "data" / "campus_frames" / "*.jpg"))):
            im = cv2.imread(f)
            if im is None:
                continue
            name = f"scene_{written:04d}"
            cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), im)
            (OUT / "labels" / f"{name}.txt").write_text("", encoding="utf-8")
            written += 1

    train_imgs = sorted((YOLO_ROOT / "images" / "train").glob("*.jpg"))
    neg_imgs = sorted((OUT / "images").glob("*.jpg"))
    LIST.write_text("\n".join(str(p) for p in train_imgs + neg_imgs), encoding="utf-8")

    import yaml

    base = yaml.safe_load((YOLO_ROOT / "data.yaml").read_text(encoding="utf-8"))
    base["path"] = str(YOLO_ROOT)
    base["train"] = str(LIST.resolve())
    YAML.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    print(f"mined {written} negative frames from {len(sources)} videos + campus frames")
    print(f"  images {OUT / 'images'}")
    print(f"  list   {LIST}  ({len(train_imgs)} train + {len(neg_imgs)} negatives)")
    print(f"  yaml   {YAML}")


if __name__ == "__main__":
    main()
