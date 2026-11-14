"""Error analysis on the test split, for both models.

Three questions the results table cannot answer:
  1. Where do the detections go wrong -- background (a box on nothing), wrong
     class (a box on real litter but the wrong label), or duplicate?
  2. Which object sizes are missed? TACO is full of small objects, so a single
     mAP number hides the fact that most of the loss is in small boxes.
  3. What do the misses look like? A grid of annotated frames is more persuasive
     in a written summary than any table.

Definitions (fixed, applied identically to both models)
  TP  : prediction with IoU >= 0.5 against an unmatched GT of the same class
  cls : prediction with IoU >= 0.5 against an unmatched GT of a *different* class
  dup : prediction matching an already-matched GT of the same class
  bg  : prediction with IoU < 0.5 against every GT  -> background / localisation
Confidence threshold for this analysis: 0.25 (a reporting choice, not a tuned
parameter; the full sweep is in the results table).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common_eval import box_iou
from data_utils import CLASSES, ROOT, load_gt
from models import load_frcnn, load_yolo, predict_frcnn, predict_yolo, read_bgr

CONF = 0.25
IOU_THR = 0.5
RESULTS = ROOT / "runs" / "results"
FIG = ROOT / "figures"
SIZE_BINS = [(0, 32, "small <32px"), (32, 96, "medium 32-96"), (96, 1e9, "large >96")]


def classify(pred: dict, gt: dict) -> tuple[Counter, list[int]]:
    counts: Counter = Counter()
    matched = torch.zeros(gt["boxes"].shape[0], dtype=torch.bool)
    keep = pred["scores"] >= CONF
    pb, pl = pred["boxes"][keep], pred["labels"][keep]
    if pb.shape[0] == 0:
        counts["missed_gt"] = int(gt["boxes"].shape[0])
        return counts, []

    ious = box_iou(pb, gt["boxes"])
    miss_idx = []
    for i in range(pb.shape[0]):
        row = ious[i]
        best_iou, best_j = torch.max(row, dim=0)
        j = int(best_j)
        if float(best_iou) < IOU_THR:
            counts["bg"] += 1
        elif int(gt["labels"][j]) != int(pl[i]):
            counts["cls"] += 1
        elif matched[j]:
            counts["dup"] += 1
        else:
            matched[j] = True
            counts["tp"] += 1
    counts["missed_gt"] = int((~matched).sum())
    miss_idx = (~matched).nonzero(as_tuple=True)[0].tolist()
    return counts, miss_idx


def size_bucket(boxes: torch.Tensor) -> list[str]:
    if boxes.shape[0] == 0:
        return []
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    side = torch.sqrt(wh[:, 0] * wh[:, 1])
    out = []
    for s in side.tolist():
        for lo, hi, name in SIZE_BINS:
            if lo <= s < hi:
                out.append(name)
                break
    return out


def analyse(name: str, preds: list[dict], gts: list[dict], paths: list[Path]) -> dict:
    totals: Counter = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    size_total: Counter = Counter()
    size_missed: Counter = Counter()
    worst: list[tuple[float, Path, dict, dict]] = []

    for p, g, path in zip(preds, gts, paths):
        c, miss = classify(p, g)
        totals.update(c)

        keep = p["scores"] >= CONF
        pb, pl = p["boxes"][keep], p["labels"][keep]
        for cls_i, cls_name in enumerate(CLASSES):
            n_gt = int((g["labels"] == cls_i).sum())
            n_pred = int((pl == cls_i).sum())
            per_class[cls_name]["gt"] += n_gt
            per_class[cls_name]["pred"] += n_pred

        buckets = size_bucket(g["boxes"])
        for b in buckets:
            size_total[b] += 1
        for i in miss:
            if i < len(buckets):
                size_missed[buckets[i]] += 1

        n_gt = g["boxes"].shape[0]
        if n_gt:
            worst.append((len(miss) / n_gt, path, p, g))

    recall_by_size = {}
    for _, _, name_b in SIZE_BINS:
        n = size_total.get(name_b, 0)
        m = size_missed.get(name_b, 0)
        recall_by_size[name_b] = {
            "n_gt": n,
            "missed": m,
            "recall": round(1 - m / n, 4) if n else None,
        }

    return {
        "model": name,
        "conf": CONF,
        "iou": IOU_THR,
        "error_counts": dict(totals),
        "recall_by_size": recall_by_size,
        "per_class_gt_pred": {k: dict(v) for k, v in per_class.items()},
        "_worst": sorted(worst, key=lambda t: -t[0])[:6],
    }


def plot(name: str, res: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    ax = axes[0]
    labels = ["TP", "background", "wrong class", "duplicate"]
    keys = ["tp", "bg", "cls", "dup"]
    vals = [res["error_counts"].get(k, 0) for k in keys]
    colors = ["#4c78a8", "#e45756", "#f58518", "#b0b0b0"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{name}: prediction outcomes @conf {CONF}")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    names = [b[2] for b in SIZE_BINS]
    rec = [res["recall_by_size"][n]["recall"] or 0 for n in names]
    ax.bar(names, rec, color="#4c78a8")
    for i, v in enumerate(rec):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{name}: recall by object size")
    ax.set_ylabel("recall")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG / f"fig_errors_{name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def grid(name: str, worst: list, n_show: int = 6) -> None:
    """Grid of the frames with the highest miss rate, GT in green, predictions in blue."""
    if not worst:
        return
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for ax, (_, path, p, g) in zip(axes.ravel(), worst[:n_show]):
        im = cv2.cvtColor(read_bgr(path), cv2.COLOR_BGR2RGB)
        for (x1, y1, x2, y2), lab in zip(g["boxes"].numpy(), g["labels"].numpy()):
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(im, CLASSES[int(lab)], (int(x1), max(10, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        keep = p["scores"] >= CONF
        for (x1, y1, x2, y2), lab in zip(p["boxes"][keep].numpy(), p["labels"][keep].numpy()):
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (30, 90, 255), 2)
        ax.imshow(im)
        ax.set_title(f"{Path(path).name}", fontsize=7)
        ax.axis("off")
    for ax in axes.ravel()[len(worst[:n_show]):]:
        ax.axis("off")
    fig.suptitle(f"{name}: green = ground truth, blue = prediction", fontsize=9)
    fig.tight_layout()
    out = FIG / f"fig_qual_{name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    records = load_gt("test")
    gts = [{"boxes": r["boxes"], "labels": r["labels"]} for r in records]
    paths = [r["path"] for r in records]

    yolo = load_yolo(ROOT / "runs" / "yolo" / "best.pt")
    yolo_preds = predict_yolo(yolo, [str(p) for p in paths])

    frcnn = load_frcnn(ROOT / "runs" / "frcnn" / "best.pt")
    frcnn_preds = predict_frcnn(frcnn, [read_bgr(p) for p in paths])

    out = {}
    for name, preds in [("YOLO11n", yolo_preds), ("Faster R-CNN", frcnn_preds)]:
        res = analyse(name, preds, gts, paths)
        worst = res.pop("_worst")
        out[name] = res
        plot(name, res)
        grid(name, worst)
        print(f"\n{name}: {res['error_counts']}")
        print(f"  recall by size: {res['recall_by_size']}")

    (RESULTS / "error_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'error_stats.json'}")


if __name__ == "__main__":
    main()
