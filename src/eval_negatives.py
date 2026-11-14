"""Did the negative frames actually help? Measure both sides of the trade.

Adding background frames can always suppress false positives -- by suppressing
everything. So this script measures three things for the original checkpoint and
the retrained one, on the same data:

1. TACO test mAP, to check the reported comparison did not degrade;
2. boxes drawn on the two self-recordings of a room, which is where the model
   was boxing hands and a drawer;
3. boxes drawn on a litter clip, to check real objects are still found.

    python src/eval_negatives.py
"""

from __future__ import annotations

import glob
import json

import cv2

from common_eval import RESULTS, evaluate_map, save_json
from data_utils import CLASSES, ROOT, load_gt, to_torchmetric_input
from live_demo import filter_pred
from models import load_yolo, predict_yolo

BASE = ROOT / "runs" / "yolo" / "best.pt"
SCENE = ROOT / "runs" / "yolo_scene" / "best.pt"
REC_DIRS = [
    ROOT / "data" / "_scene_frames" / "rec1",
    ROOT / "data" / "_scene_frames" / "rec2",
]
LITTER_CLIP = ROOT / "data" / "_scene_frames" / "litter"


def count_boxes(model, paths, conf: float = 0.60) -> int:
    if not paths:
        return 0
    preds = predict_yolo(model, paths, conf=0.05)
    n = 0
    for p, d in zip(paths, preds):
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        f = filter_pred(d, conf, [0, 1, 2], 4.0, 0.0005, float(h * w), 0)
        n += len(f["scores"])
    return n


def main() -> None:
    gts = to_torchmetric_input(load_gt("test"))
    test_paths = [str(r["path"]) for r in load_gt("test")]

    recs = sorted(glob.glob(str(REC_DIRS[0] / "*.jpg"))) + sorted(glob.glob(str(REC_DIRS[1] / "*.jpg")))
    litter = sorted(glob.glob(str(LITTER_CLIP / "*.jpg")))
    if not recs or not litter:
        raise SystemExit(
            "missing frame sets. Decode them first:\n"
            "  ffmpeg -i demo/raw/<clip>.mp4 -vf fps=1,scale=640:-2 data/_scene_frames/recN/%04d.jpg\n"
            "  ffmpeg -i demo/test_clip_synthetic_v2.mp4 -vf fps=2 data/_scene_frames/litter/%04d.jpg"
        )

    out = {"checkpoints": {}, "n_frames": {"recordings": len(recs), "litter": len(litter)}}
    for tag, w in (("original", BASE), ("with_negatives", SCENE)):
        if not w.exists():
            raise SystemExit(f"missing {w}; run mine_scene_negatives.py then train_yolo.py")
        m = load_yolo(w)
        r = evaluate_map(predict_yolo(m, test_paths), gts)
        out["checkpoints"][tag] = {
            "weights": str(w.relative_to(ROOT)),
            "mAP_0.5": r["mAP_0.5"],
            "mAP_0.5:0.95": r["mAP_0.5:0.95"],
            "boxes_on_recordings": count_boxes(m, recs),
            "boxes_on_litter_clip": count_boxes(m, litter),
        }
        c = out["checkpoints"][tag]
        print(f"{tag:<15} mAP50 {c['mAP_0.5']:.4f}  mAP {c['mAP_0.5:0.95']:.4f}  "
              f"recordings {c['boxes_on_recordings']} boxes  litter {c['boxes_on_litter_clip']} boxes")

    b, s = out["checkpoints"]["original"], out["checkpoints"]["with_negatives"]
    out["delta"] = {
        "mAP_0.5": round(s["mAP_0.5"] - b["mAP_0.5"], 4),
        "boxes_on_recordings": s["boxes_on_recordings"] - b["boxes_on_recordings"],
    }
    save_json(out, "scene_negatives.json")
    print(f"\nwrote {RESULTS / 'scene_negatives.json'}")


if __name__ == "__main__":
    main()
