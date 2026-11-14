"""What the demo configuration costs, measured on the test split.

`live_demo.py --demo-safe` trades recall for precision: it raises the confidence
floor to 0.60 and drops the two classes whose heads never trained (cigarette
AP@0.5:0.95 = 0.0039, glass = 0.0002-0.0079). That is a deliberate choice for a
live demonstration, and a deliberate choice has to be priced.

This script re-scores the saved test predictions under each candidate
configuration, so the README can state the cost instead of hand-waving it.

Not included: the geometric filters (ROI band, aspect ratio, minimum area,
two-frame stability). Those depend on a camera geometry that the test split does
not share, so scoring them here would be inventing a number. They are demo-time
constraints only, and the README says so.

    python src/deploy_config.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from common_eval import RESULTS, greedy_pr, save_json
from data_utils import CLASSES, ROOT

SAFE_CLASSES = [0, 1, 2]  # plastic, paper_cardboard, metal

# (label, classes kept, confidence floor)
CONFIGS: list[tuple[str, list[int] | None, float]] = [
    ("research · 6 classes · conf 0.25", None, 0.25),
    ("research · 6 classes · conf 0.60", None, 0.60),
    # same class set, low threshold: the only apples-to-apples row for recall,
    # because dropping dead classes also removes them from the recall denominator
    ("research · 3 classes · conf 0.25", SAFE_CLASSES, 0.25),
    ("demo-safe · 3 classes · conf 0.40", SAFE_CLASSES, 0.40),
    ("demo-safe · 3 classes · conf 0.55", SAFE_CLASSES, 0.55),
    ("demo-safe · 3 classes · conf 0.60", SAFE_CLASSES, 0.60),
    ("demo-safe · 3 classes · conf 0.70", SAFE_CLASSES, 0.70),
]


def load(split: str = "test") -> tuple[dict, dict]:
    from data_utils import load_gt, to_torchmetric_input

    records = load_gt(split)
    gts = to_torchmetric_input(records)
    preds = {
        "YOLO11n": torch.load(RESULTS / f"preds_YOLO_{split}.pt", weights_only=False),
        "FasterRCNN": torch.load(RESULTS / f"preds_FasterRCNN_{split}.pt", weights_only=False),
    }
    return preds, gts


def filter_preds(preds: list[dict], classes: list[int] | None, conf: float) -> list[dict]:
    out = []
    for p in preds:
        boxes, scores, labels = p["boxes"], p["scores"], p["labels"]
        keep = scores >= conf
        if classes is not None:
            keep = keep & torch.isin(labels, torch.tensor(classes, dtype=labels.dtype))
        out.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    return out


def filter_gts(gts: list[dict], classes: list[int] | None) -> list[dict]:
    if classes is None:
        return gts
    out = []
    for g in gts:
        keep = torch.isin(g["labels"], torch.tensor(classes, dtype=g["labels"].dtype))
        out.append({"boxes": g["boxes"][keep], "labels": g["labels"][keep]})
    return out


def main() -> None:
    preds, gts = load("test")
    out: dict = {"split": "test", "classes": CLASSES, "configs": []}

    for label, classes, conf in CONFIGS:
        row = {"config": label, "classes_kept": classes or list(range(len(CLASSES))), "conf": conf}
        gt_f = filter_gts(gts, classes)
        for model in ("YOLO11n", "FasterRCNN"):
            p_f = filter_preds(preds[model], classes, conf)
            pr = greedy_pr(p_f, gt_f, conf=0.0)  # already filtered above
            row[model] = {
                "precision": pr["precision"],
                "recall": pr["recall"],
                "f1": pr["f1"],
                "tp": pr["tp"],
                "fp": pr["fp"],
                "fn": pr["fn"],
                "n_det": int(sum(len(p["scores"]) for p in p_f)),
            }
        out["configs"].append(row)
        print(
            f"{label:<36} YOLO P={row['YOLO11n']['precision']:.4f} "
            f"R={row['YOLO11n']['recall']:.4f} | R-CNN P={row['FasterRCNN']['precision']:.4f} "
            f"R={row['FasterRCNN']['recall']:.4f}"
        )

    save_json(out, "deploy_config.json")

    lines = [
        "# Deployment configuration: what the demo costs",
        "",
        "Re-scored from the saved test predictions. The geometric demo filters "
        "(ROI band, aspect ratio, minimum area, stability) are **not** included -- "
        "they depend on camera geometry the test split does not share.",
        "",
        "| configuration | YOLO P | YOLO R | YOLO F1 | R-CNN P | R-CNN R | R-CNN F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in out["configs"]:
        y, f = row["YOLO11n"], row["FasterRCNN"]
        lines.append(
            f"| {row['config']} | {y['precision']} | {y['recall']} | {y['f1']} | "
            f"{f['precision']} | {f['recall']} | {f['f1']} |"
        )
    lines += [
        "",
        "Reading it: moving from the research setting to the demo setting raises "
        "precision (fewer junk boxes on screen) and lowers recall (fewer real "
        "objects found). Both are reported; neither is hidden.",
        "",
    ]
    (RESULTS / "deploy_config.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'deploy_config.md'}")


if __name__ == "__main__":
    main()
