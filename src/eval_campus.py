"""Out-of-domain evaluation on the campus frame set.

Two modes, chosen automatically from the annotation file:

**Positive mode** (the set has boxes) -- the TACO test mAP says how well a model
does on photos of litter taken by strangers, mostly on beaches, trails and
roadsides. It does not say how the same model behaves on a campus walkway at
midday through a webcam. This mode measures that gap with the same metric code,
the same input size and the same NMS settings used everywhere else, so the two
numbers are directly comparable.

**Negative mode** (the set has no boxes at all) -- a set of ordinary scenes with
no litter in them. Every detection on such a frame is by definition a false
positive, so the meaningful number is FP per frame, not mAP. This is the mode
that answers "does it hallucinate litter on a blackboard", which is a failure
the in-domain test split cannot show, because every image in TACO contains
litter.

Both modes require the frames to be flagged `reviewed: true`. `preannotate.py`
seeds boxes with the model under test, so an unreviewed frame is a model grading
its own homework: measured on 30 such frames, YOLO scored mAP@0.5 = 1.0 against
itself. Filtering on the flag is the difference between a measurement and a
tautology. For negative mode the human review is trivial -- confirm the frame
really contains no litter -- but it still has to happen.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

from common_eval import evaluate_map, greedy_pr, save_json
from data_utils import CLASSES, ROOT
from models import load_frcnn, load_yolo, predict_frcnn, predict_yolo, read_bgr

CAMPUS = Path(os.environ.get("CAMPUS_DIR", str(ROOT / "data" / "campus")))
RESULTS = ROOT / "runs" / "results"
CONF_SWEEP = [0.10, 0.25, 0.40, 0.55, 0.70]
REPORT_CONF = 0.25


def load_campus() -> tuple[list[dict], int, int]:
    """Returns (records for hand-reviewed frames, frames excluded, annotation count)."""
    p = CAMPUS / "_annotations.coco.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n"
            "  1) python src/live_demo.py --record        # walk, press r\n"
            "  2) python src/extract_frames.py <clip> --fps 2\n"
            "  3) mark every frame reviewed=true (boxes may stay empty for a "
            "litter-free scene)\n"
            "  4) re-run this script"
        )
    data = json.loads(p.read_text(encoding="utf-8"))

    # A frame only counts once a human has marked it reviewed. See module docstring.
    reviewed = [img for img in data["images"] if img.get("reviewed") is True]
    excluded = len(data["images"]) - len(reviewed)
    if not reviewed:
        raise SystemExit(
            "no frame is marked reviewed=true, so there is nothing to score.\n"
            "  The boxes in this set were seeded by the model under test; scoring them\n"
            "  would be circular and would report a meaningless number.\n"
            "  1) correct the boxes by hand in any COCO editor (CVAT, Label Studio, ...)\n"
            "     -- or delete them all, if the scene genuinely contains no litter\n"
            "  2) set \"reviewed\": true on every frame you actually checked\n"
            "  3) re-run this script"
        )

    per_image = defaultdict(list)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        per_image[ann["image_id"]].append((x, y, x + w, y + h, ann["category_id"]))

    records = []
    for img in reviewed:
        items = per_image.get(img["id"], [])
        records.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "path": CAMPUS / "images" / img["file_name"],
                "width": img["width"],
                "height": img["height"],
                "boxes": torch.tensor([it[:4] for it in items], dtype=torch.float32)
                if items else torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.tensor([it[4] for it in items], dtype=torch.int64)
                if items else torch.zeros((0,), dtype=torch.int64),
            }
        )
    return records, excluded, len(data["annotations"])


def _negative_stats(preds: list[dict], n_images: int) -> dict:
    """Every detection on a litter-free frame is a false positive."""
    out: dict = {"fp_per_frame": {}, "total": {}, "by_class_at_report_conf": {}}
    for c in CONF_SWEEP:
        total = int(sum((p["scores"].numpy() >= c).sum() for p in preds))
        out["total"][f"{c:.2f}"] = total
        out["fp_per_frame"][f"{c:.2f}"] = round(total / max(n_images, 1), 3)

    counts: Counter = Counter()
    for p in preds:
        scores = p["scores"].numpy()
        labels = p["labels"].numpy()
        for lab in labels[scores >= REPORT_CONF]:
            counts[CLASSES[int(lab)]] += 1
    out["by_class_at_report_conf"] = dict(counts)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="campus", help="suffix for the result files")
    args = ap.parse_args()

    records, excluded, n_ann = load_campus()
    gts = [{"boxes": r["boxes"], "labels": r["labels"]} for r in records]
    n_inst = int(sum(len(g["labels"]) for g in gts))
    negative = n_inst == 0

    mode = "negative" if negative else "positive"
    print(f"campus set: {len(records)} reviewed images, {n_inst} instances -> {mode} mode")
    if excluded:
        print(f"  ({excluded} unreviewed frames excluded -- they would be self-scored)")
    if negative and n_ann:
        print(
            f"  note: the annotation file holds {n_ann} box(es) but none belong to a "
            "reviewed frame; treating the set as litter-free"
        )

    out = {
        "mode": mode,
        "n_images": len(records),
        "n_instances": n_inst,
        "n_excluded_unreviewed": excluded,
        "classes": CLASSES,
        "models": {},
    }

    yolo = load_yolo(ROOT / "runs" / "yolo" / "best.pt")
    yp = predict_yolo(yolo, [str(r["path"]) for r in records])

    frcnn = load_frcnn(ROOT / "runs" / "frcnn" / "best.pt")
    fp = predict_frcnn(frcnn, [read_bgr(r["path"]) for r in records])

    if negative:
        for name, preds in (("YOLO11n", yp), ("FasterRCNN", fp)):
            out["models"][name] = _negative_stats(preds, len(records))
            s = out["models"][name]
            print(
                f"{name:<14} FP/frame {s['fp_per_frame'][f'{REPORT_CONF:.2f}']} at conf "
                f"{REPORT_CONF:.2f}; {s['total']['0.55']} FP total at conf 0.55"
            )
    else:
        out["models"]["YOLO11n"] = {
            "map": evaluate_map(yp, gts),
            "pr": {f"{c:.2f}": greedy_pr(yp, gts, c) for c in CONF_SWEEP},
        }
        print(f"YOLO   mAP@0.5={out['models']['YOLO11n']['map']['mAP_0.5']}")
        out["models"]["FasterRCNN"] = {
            "map": evaluate_map(fp, gts),
            "pr": {f"{c:.2f}": greedy_pr(fp, gts, c) for c in CONF_SWEEP},
        }
        print(f"R-CNN  mAP@0.5={out['models']['FasterRCNN']['map']['mAP_0.5']}")

    save_json(out, f"eval_campus_{args.tag}.json")
    if not negative and args.tag == "campus":
        save_json(out, "eval_campus.json")

    lines = [
        f"# Domain shift — {mode} mode ({len(records)} frames)",
        "",
    ]
    if negative:
        lines += [
            "Litter-free frames. Every detection is a false positive; the metric is "
            "false positives per frame, not mAP.",
            "",
            "| model | conf 0.25 | conf 0.40 | conf 0.55 | conf 0.70 |",
            "|---|---|---|---|---|",
        ]
        for name in ("YOLO11n", "FasterRCNN"):
            s = out["models"][name]
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    name,
                    s["fp_per_frame"]["0.25"],
                    s["fp_per_frame"]["0.40"],
                    s["fp_per_frame"]["0.55"],
                    s["fp_per_frame"]["0.70"],
                )
            )
        lines += ["", f"Per class at conf {REPORT_CONF:.2f}:", "", "| model | " +
                  " | ".join(CLASSES) + " |", "|---" * (len(CLASSES) + 1) + "|"]
        for name in ("YOLO11n", "FasterRCNN"):
            bc = out["models"][name]["by_class_at_report_conf"]
            lines.append(
                f"| {name} | " + " | ".join(str(bc.get(c, 0)) for c in CLASSES) + " |"
            )
    else:
        test_p = RESULTS / "eval_test.json"
        lines += [
            "| mAP@0.5 | YOLO11n (TACO test) | YOLO11n (campus) | R-CNN (TACO test) | R-CNN (campus) |",
            "|---|---|---|---|---|",
        ]
        if test_p.exists():
            t = json.loads(test_p.read_text(encoding="utf-8"))["models"]
            for k, label in [("mAP_0.5", "mAP@0.5"), ("mAP_0.5:0.95", "mAP@0.5:0.95")]:
                ty, tc = t["YOLO11n"]["map"][k], t["FasterRCNN"]["map"][k]
                cy, cc = out["models"]["YOLO11n"]["map"][k], out["models"]["FasterRCNN"]["map"][k]
                lines.append(
                    f"| {label} | {ty} | {cy} ({round(cy - ty, 4):+}) | {tc} | "
                    f"{cc} ({round(cc - tc, 4):+}) |"
                )
        else:
            lines.append("| (run run_eval.py first for the in-domain column) | | | | |")

    (RESULTS / f"domain_shift_{args.tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not negative and args.tag == "campus":
        (RESULTS / "domain_shift.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
