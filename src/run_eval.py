"""Evaluate both models on a prepared split with ONE implementation.

The test split is touched exactly once, here. Nothing in this file is tuned on
the test split: the confidence sweep is reported, not used to pick a winner.

Usage
-----
    python src/run_eval.py                # TACO test split
    python src/run_eval.py --split valid  # validation split (sanity check)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from common_eval import evaluate_map, greedy_pr, save_json
from data_utils import CLASSES, ROOT, load_gt, to_torchmetric_input
from models import load_frcnn, load_yolo, predict_frcnn, predict_yolo, read_bgr, to_eval_dicts

CONF_SWEEP = [0.10, 0.25, 0.40, 0.55, 0.70]
RESULTS = ROOT / "runs" / "results"


def run_split(split: str, tag: str | None = None) -> dict:
    tag = tag or split
    records = load_gt(split)
    gts = to_torchmetric_input(records)
    paths = [r["path"] for r in records]
    print(f"[{tag}] {len(records)} images, {sum(len(g['labels']) for g in gts)} instances")

    yolo_w = ROOT / "runs" / "yolo" / "best.pt"
    frcnn_w = Path(os.environ.get("FRCNN_WEIGHTS", str(ROOT / "runs" / "frcnn" / "best.pt")))
    for w in (yolo_w, frcnn_w):
        if not w.exists():
            raise SystemExit(f"missing weights: {w}")

    summary: dict[str, dict] = {}

    # ---------------- YOLO ----------------
    print(f"[{tag}] YOLO ...", flush=True)
    yolo = load_yolo(yolo_w)
    t0 = time.time()
    preds = predict_yolo(yolo, [str(p) for p in paths])
    print(f"[{tag}] YOLO inference {time.time() - t0:.1f}s")
    torch.save([{k: v for k, v in p.items()} for p in preds], RESULTS / f"preds_YOLO_{tag}.pt")

    yolo_map = evaluate_map(preds, gts)
    yolo_pr = {f"{c:.2f}": greedy_pr(preds, gts, c) for c in CONF_SWEEP}
    summary["YOLO11n"] = {"map": yolo_map, "pr": yolo_pr}
    print(f"[{tag}] YOLO mAP@0.5={yolo_map['mAP_0.5']}  mAP@0.5:0.95={yolo_map['mAP_0.5:0.95']}")

    # ---------------- Faster R-CNN ----------------
    print(f"[{tag}] Faster R-CNN ...", flush=True)
    frcnn = load_frcnn(frcnn_w)
    t0 = time.time()
    frames = [read_bgr(p) for p in paths]
    frcnn_preds = predict_frcnn(frcnn, frames)
    print(f"[{tag}] Faster R-CNN inference {time.time() - t0:.1f}s")
    torch.save(frcnn_preds, RESULTS / f"preds_FasterRCNN_{tag}.pt")

    frcnn_map = evaluate_map(frcnn_preds, gts)
    frcnn_pr = {f"{c:.2f}": greedy_pr(frcnn_preds, gts, c) for c in CONF_SWEEP}
    summary["FasterRCNN"] = {"map": frcnn_map, "pr": frcnn_pr}
    print(
        f"[{tag}] R-CNN mAP@0.5={frcnn_map['mAP_0.5']}  "
        f"mAP@0.5:0.95={frcnn_map['mAP_0.5:0.95']}"
    )

    out = {
        "split": split,
        "tag": tag,
        "n_images": len(records),
        "n_instances": int(sum(len(g["labels"]) for g in gts)),
        "classes": CLASSES,
        "models": summary,
    }
    save_json(out, f"eval_{tag}.json")
    write_table(out, tag)
    return out


def write_table(out: dict, tag: str) -> Path:
    y, f = out["models"]["YOLO11n"], out["models"]["FasterRCNN"]
    lines = [
        f"# Results — {tag} ({out['n_images']} images, {out['n_instances']} instances)",
        "",
        "| Metric | YOLO11n | Faster R-CNN |",
        "|---|---|---|",
    ]
    for key, label in [
        ("mAP_0.5:0.95", "mAP@0.5:0.95"),
        ("mAP_0.5", "mAP@0.5"),
        ("mAP_0.75", "mAP@0.75"),
        ("mAP_small", "mAP small"),
        ("mAP_medium", "mAP medium"),
        ("mAP_large", "mAP large"),
    ]:
        lines.append(f"| {label} | {y['map'][key]} | {f['map'][key]} |")

    lines += ["", "## Per-class AP@0.5:0.95", "", "| Class | YOLO11n | Faster R-CNN |", "|---|---|---|"]
    for c in CLASSES:
        lines.append(
            f"| {c} | {y['map']['per_class_AP'].get(c, '-')} | {f['map']['per_class_AP'].get(c, '-')} |"
        )

    lines += [
        "",
        "## Precision / Recall / F1 (IoU 0.5)",
        "",
        "| conf | YOLO P | YOLO R | YOLO F1 | R-CNN P | R-CNN R | R-CNN F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in y["pr"]:
        a, b = y["pr"][c], f["pr"][c]
        lines.append(
            f"| {c} | {a['precision']} | {a['recall']} | {a['f1']} | "
            f"{b['precision']} | {b['recall']} | {b['f1']} |"
        )

    p = RESULTS / f"results_{tag}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(p.read_text(encoding="utf-8"))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    run_split(args.split, args.tag)


if __name__ == "__main__":
    main()
