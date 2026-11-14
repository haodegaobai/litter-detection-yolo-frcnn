"""Exploratory data analysis: three figures used by this project.

fig_class_distribution -- instances per superclass, train/valid/test stacked. The
    point of this figure is the imbalance: plastic is 4x cigarette and 8x glass,
    which explains most of the per-class AP spread later.
fig_box_size_distribution -- box side length on a log scale, with the 32 px and
    96 px COCO small/medium boundaries drawn in. Most TACO boxes are small, which
    is the single biggest driver of mAP in the low 0.2s.
fig_gt_samples -- a 3x3 grid of ground truth, so the reader can see what the
    annotations look like before reading any number.
"""

from __future__ import annotations

import json
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import CLASSES, ROOT, load_gt

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]


def fig_classes(summary: dict) -> None:
    splits = ["train", "valid", "test"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = np.zeros(len(CLASSES))
    for i, s in enumerate(splits):
        vals = np.array([summary["splits"][s]["per_class"][c] for c in CLASSES], dtype=float)
        ax.bar(CLASSES, vals, bottom=bottom, label=s, color=["#2f5f8f", "#5f9fd0", "#a8cbe8"][i])
        bottom += vals
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("instances")
    ax.set_title("Instances per superclass (TACO folded from 60 categories)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_class_distribution.png", dpi=180)
    print("fig1_class_distribution.png")


def fig_box_sizes() -> None:
    sides = []
    for split in ("train", "valid", "test"):
        for r in load_gt(split):
            if r["boxes"].shape[0]:
                wh = (r["boxes"][:, 2:] - r["boxes"][:, :2]).clamp(min=0)
                sides.extend(np.sqrt((wh[:, 0] * wh[:, 1]).numpy()).tolist())

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.hist(sides, bins=np.logspace(0, 3.2, 60), color="#4c78a8", alpha=0.85)
    ax.set_xscale("log")
    for x, label in [(32, "COCO small/medium = 32 px"), (96, "medium/large = 96 px")]:
        ax.axvline(x, color="#e45756", ls="--", lw=1)
        ax.text(x * 1.05, ax.get_ylim()[1] * 0.92, label, fontsize=7, color="#e45756")
    ax.set_xlabel("box side (px, geometric mean) — log scale")
    ax.set_ylabel("count")
    small = sum(1 for s in sides if s < 32)
    ax.set_title(
        f"Object size distribution — {small / len(sides) * 100:.0f}% of boxes are <32 px "
        f"(n={len(sides)})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_box_size_distribution.png", dpi=180)
    print("fig2_box_size_distribution.png")


def fig_samples(n: int = 9) -> None:
    recs = [r for r in load_gt("train") if r["boxes"].shape[0] >= 2][:n]
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 7.2))
    for ax, r in zip(axes.ravel(), recs):
        im = cv2.cvtColor(cv2.imread(str(r["path"])), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        scale = 1.0
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            im = cv2.resize(im, (int(w * scale), int(h * scale)))
        for (x1, y1, x2, y2), lab in zip(r["boxes"].numpy(), r["labels"].numpy()):
            x1, y1, x2, y2 = [v * scale for v in (x1, y1, x2, y2)]
            c = COLORS[int(lab) % len(COLORS)]
            rgb = tuple(int(c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), rgb, 2)
            cv2.putText(im, CLASSES[int(lab)], (int(x1), max(9, int(y1) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, rgb, 1)
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("Ground truth samples (folded classes)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_gt_samples.png", dpi=150)
    print("fig3_gt_samples.png")


def main() -> None:
    summary = json.loads((ROOT / "data" / "dataset_summary.json").read_text(encoding="utf-8"))
    print("totals:", summary["totals"])
    for s in ("train", "valid", "test"):
        d = summary["splits"][s]
        print(f"  {s:>6}: {d['images']:>4} images {d['objects']:>5} objects {d['per_class']}")
    fig_classes(summary)
    fig_box_sizes()
    fig_samples()


if __name__ == "__main__":
    main()
