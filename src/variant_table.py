"""One table for the promotion decision: every candidate checkpoint on the same three questions.

The delivered version must not move unless a candidate is better on the questions
that matter, so this script measures all of them in one pass:

1. TACO test mAP -- the headline number;
2. false positives per frame on 55 confirmed litter-free campus frames;
3. boxes on a litter clip -- a sanity check that a candidate has not simply
   learned to say nothing.

    python src/variant_table.py
"""

from __future__ import annotations

import glob

import cv2

from common_eval import RESULTS, evaluate_map, save_json
from data_utils import ROOT, load_gt, to_torchmetric_input
from live_demo import filter_pred
from models import load_yolo, predict_yolo

CANDIDATES = [
    ("reported · seed 42", ROOT / "runs" / "yolo" / "best.pt", 640),
    ("seed 7", ROOT / "runs" / "yolo_seed7" / "best.pt", 640),
    ("+ negatives", ROOT / "runs" / "yolo_scene" / "best.pt", 640),
    ("yolo11s", ROOT / "runs" / "yolo11s" / "best.pt", 640),
]
CAMPUS = ROOT / "data" / "campus_frames"
LITTER = ROOT / "data" / "_scene_frames" / "litter"


def boxes_on(model, paths, conf: float = 0.60, classes=(0, 1, 2)) -> int:
    if not paths:
        return -1
    preds = predict_yolo(model, paths, conf=0.05)
    n = 0
    for p, d in zip(paths, preds):
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        f = filter_pred(d, conf, list(classes), 4.0, 0.0005, float(h * w), 0)
        n += len(f["scores"])
    return n


def main() -> None:
    gts = to_torchmetric_input(load_gt("test"))
    test_paths = [str(r["path"]) for r in load_gt("test")]
    campus = sorted(glob.glob(str(CAMPUS / "*.jpg")))
    litter = sorted(glob.glob(str(LITTER / "*.jpg")))

    rows = []
    for name, w, imgsz in CANDIDATES:
        if not w.exists():
            print(f"skip {name}: {w} missing")
            continue
        m = load_yolo(w)
        r = evaluate_map(predict_yolo(m, test_paths), gts)
        fp_campus = boxes_on(m, campus)
        fp_litter = boxes_on(m, litter)
        rows.append({
            "variant": name,
            "weights": str(w.relative_to(ROOT)),
            "mAP_0.5": r["mAP_0.5"],
            "mAP_0.5:0.95": r["mAP_0.5:0.95"],
            "fp_per_frame_campus": round(fp_campus / max(len(campus), 1), 3),
            "boxes_on_litter_clip": fp_litter,
        })
        x = rows[-1]
        print(f"{name:<20} mAP50 {x['mAP_0.5']:.4f}  mAP {x['mAP_0.5:0.95']:.4f}  "
              f"campus FP/frame {x['fp_per_frame_campus']}  litter boxes {x['boxes_on_litter_clip']}")

    if rows:
        base = rows[0]
        for x in rows:
            x["d_mAP_0.5"] = round(x["mAP_0.5"] - base["mAP_0.5"], 4)
        save_json({"campus_frames": len(campus), "litter_frames": len(litter), "variants": rows},
                  "variant_table.json")

        lines = ["# Candidate checkpoints", "",
                 f"{len(campus)} litter-free campus frames, {len(litter)} litter frames, TACO test split.", "",
                 "| variant | mAP@0.5 | delta | mAP@0.5:0.95 | campus FP/frame | litter boxes |",
                 "|---|---|---|---|---|---|"]
        for x in rows:
            lines.append(f"| {x['variant']} | {x['mAP_0.5']} | {x['d_mAP_0.5']:+} | "
                         f"{x['mAP_0.5:0.95']} | {x['fp_per_frame_campus']} | {x['boxes_on_litter_clip']} |")
        lines += ["", "Delta is against the reported checkpoint. Nothing here is promoted automatically.", ""]
        (RESULTS / "variant_table.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nwrote {RESULTS / 'variant_table.md'}")


if __name__ == "__main__":
    main()
