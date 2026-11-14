"""The single evaluation path used by BOTH models.

Every number in the results table comes out of this module. YOLO's own `val()`
mAP is deliberately NOT used: it is computed by a different implementation
(different interpolation, different matching) and mixing the two would make the
comparison meaningless.

Conventions
-----------
* boxes: absolute pixel xyxy in the image frame the model was given
* labels: 0-based, indices into CLASSES
* mAP: torchmetrics MeanAveragePrecision (COCO-style 101-point interpolation)
* Precision / Recall: greedy matching at IoU 0.5 and a fixed confidence
  threshold, micro-averaged over classes unless stated otherwise
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torchmetrics.detection import MeanAveragePrecision

from data_utils import CLASSES, NUM_CLASSES, ROOT

RESULTS = ROOT / "runs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU matrix between (N,4) and (M,4) xyxy boxes."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]))
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9)


def evaluate_map(preds: list[dict], gts: list[dict]) -> dict:
    """COCO-style mAP with torchmetrics. Returns overall + per-class AP."""
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox", class_metrics=True, extended_summary=False
    )
    metric.update(
        [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in preds],
        [{"boxes": g["boxes"], "labels": g["labels"]} for g in gts],
    )
    r = metric.compute()

    per_class = {}
    mp = r.get("map_per_class")
    classes = r.get("classes")
    if mp is not None and classes is not None:
        for cls_id, ap in zip(classes.tolist(), mp.tolist()):
            idx = int(cls_id)
            if 0 <= idx < len(CLASSES):
                per_class[CLASSES[idx]] = round(float(ap), 4)

    return {
        "mAP_0.5:0.95": round(float(r["map"]), 4),
        "mAP_0.5": round(float(r["map_50"]), 4),
        "mAP_0.75": round(float(r["map_75"]), 4),
        "mAP_small": round(float(r["map_small"]), 4) if "map_small" in r else None,
        "mAP_medium": round(float(r["map_medium"]), 4) if "map_medium" in r else None,
        "mAP_large": round(float(r["map_large"]), 4) if "map_large" in r else None,
        "per_class_AP": per_class,
    }


def greedy_pr(preds: list[dict], gts: list[dict], conf: float, iou_thr: float = 0.5) -> dict:
    """Precision / recall via greedy matching, per class and micro-averaged."""
    tp = [0] * NUM_CLASSES
    fp = [0] * NUM_CLASSES
    n_gt = [0] * NUM_CLASSES

    for p, g in zip(preds, gts):
        gb, gl = g["boxes"], g["labels"]
        for c in range(NUM_CLASSES):
            n_gt[c] += int((gl == c).sum())

        keep = p["scores"] >= conf
        pb, pl = p["boxes"][keep], p["labels"][keep]
        if pb.shape[0] == 0:
            continue

        for c in range(NUM_CLASSES):
            sel = pl == c
            if int(sel.sum()) == 0:
                continue
            b = pb[sel]
            gb_c = gb[gl == c]
            if gb_c.shape[0] == 0:
                fp[c] += b.shape[0]
                continue

            ious = box_iou(b, gb_c)
            matched = torch.zeros(gb_c.shape[0], dtype=torch.bool)
            for i in range(b.shape[0]):
                row = ious[i].masked_fill(matched, -1.0)
                best_iou, best_j = torch.max(row, dim=0)
                if float(best_iou) >= iou_thr:
                    matched[int(best_j)] = True
                    tp[c] += 1
                else:
                    fp[c] += 1

    eps = 1e-9
    per_class = {}
    for c in range(NUM_CLASSES):
        per_class[CLASSES[c]] = {
            "precision": round(tp[c] / (tp[c] + fp[c] + eps), 4),
            "recall": round(tp[c] / (n_gt[c] + eps), 4),
            "tp": tp[c],
            "fp": fp[c],
            "n_gt": n_gt[c],
        }

    TP, FP, NG = sum(tp), sum(fp), sum(n_gt)
    micro_p = TP / (TP + FP + eps)
    micro_r = TP / (NG + eps)
    f1 = 2 * micro_p * micro_r / (micro_p + micro_r + eps)
    return {
        "conf": conf,
        "iou": iou_thr,
        "precision": round(micro_p, 4),
        "recall": round(micro_r, 4),
        "f1": round(f1, 4),
        "tp": TP,
        "fp": FP,
        "fn": NG - TP,
        "per_class": per_class,
    }


def save_json(obj, name: str) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return p
