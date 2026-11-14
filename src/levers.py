"""Measure the three levers that could push this model further, without touching the main comparison.

The headline result is YOLO11n against Faster R-CNN, both at 640 px on
the same split. That comparison stays as it is. This script answers a different
question -- "what else would it take to be better?" -- by scoring variants on the
same test split with the same metric code:

    baseline     yolo11n @ 640          (the reported model)
    resolution   yolo11n fine-tuned @ 960
    negatives    yolo11n retrained with 400 mined background patches
    capacity     yolo11s @ 640

Each variant is timed with the benchmark protocol (batch 1, FP32, 10 warm-up
discarded, 100 timed, CUDA sync) so the cost of the gain is visible next to it.

    python src/levers.py
"""

from __future__ import annotations

import json
import time

import torch

import models as M
from benchmark import timed
from common_eval import RESULTS, evaluate_map, greedy_pr, save_json
from data_utils import ROOT, load_gt, to_torchmetric_input

VARIANTS = [
    ("baseline · yolo11n · 640", ROOT / "runs" / "yolo" / "best.pt", 640),
    ("resolution · yolo11n · 960", ROOT / "runs" / "yolo960" / "best.pt", 960),
    ("negatives · yolo11n · 640", ROOT / "runs" / "yolo_neg" / "best.pt", 640),
    ("capacity · yolo11s · 640", ROOT / "runs" / "yolo11s" / "best.pt", 640),
]


def make_fn(model, imgsz: int):
    """Wrap a model so benchmark.timed can drive it at a chosen input size."""

    def fn(frame):
        old = M.IMGSZ
        M.IMGSZ = imgsz
        try:
            return M.predict_yolo(model, [frame])
        finally:
            M.IMGSZ = old

    return fn


def main() -> None:
    records = load_gt("test")
    gts = to_torchmetric_input(records)
    paths = [str(r["path"]) for r in records]
    frames = [M.read_bgr(r["path"]) for r in records]

    rows = []
    for label, weights, imgsz in VARIANTS:
        if not weights.exists():
            print(f"skip {label}: {weights} missing")
            continue
        model = M.load_yolo(weights)
        old = M.IMGSZ
        M.IMGSZ = imgsz
        try:
            # chunk the source list: ultralytics batches internally and 225 images
            # at 960 px does not fit in 6 GB
            # 960 does not fit in 6 GB unless chunked; at 640 use one call so the
            # baseline row reproduces the headline result exactly (batching changes which
            # convolution kernels cudnn picks, which can flip borderline detections)
            chunk = 4 if imgsz >= 960 else len(paths)
            preds = []
            for i in range(0, len(paths), chunk):
                preds.extend(M.predict_yolo(model, paths[i:i + chunk]))
                torch.cuda.empty_cache()
        finally:
            M.IMGSZ = old

        m = evaluate_map(preds, gts)
        pr = greedy_pr(preds, gts, conf=0.25)
        pr50 = greedy_pr(preds, gts, conf=0.50)
        timing = timed(make_fn(model, imgsz), frames[:32])
        params = sum(p.numel() for p in model.model.parameters())

        row = {
            "variant": label,
            "imgsz": imgsz,
            "params": params,
            "mAP_0.5": m["mAP_0.5"],
            "mAP_0.5:0.95": m["mAP_0.5:0.95"],
            "mAP_small": m["mAP_small"],
            "precision_0.25": pr["precision"],
            "recall_0.25": pr["recall"],
            "fp_at_0.50": pr50["fp"],
            "mean_ms": timing["mean_ms"],
            "fps": timing["fps"],
        }
        rows.append(row)
        del model
        torch.cuda.empty_cache()
        print(
            f"{label:<30} mAP50 {row['mAP_0.5']:.4f}  mAP {row['mAP_0.5:0.95']:.4f}  "
            f"small {row['mAP_small']}  {row['mean_ms']:.2f} ms  FP@0.5 {row['fp_at_0.50']}"
        )

    if not rows:
        raise SystemExit("no variant weights found; train one first")

    base = next((r for r in rows if r["variant"].startswith("baseline")), rows[0])
    for r in rows:
        r["d_mAP_0.5"] = round(r["mAP_0.5"] - base["mAP_0.5"], 4)
        r["d_mAP_small"] = (round(r["mAP_small"] - base["mAP_small"], 4)
                            if r["mAP_small"] is not None and base["mAP_small"] is not None else None)
        r["speed_ratio"] = round(r["mean_ms"] / base["mean_ms"], 2)

    save_json({"split": "test", "baseline": base["variant"], "variants": rows}, "levers.json")

    lines = [
        "# Three levers, measured on the test split",
        "",
        "Same split, same metric code, same timing protocol. The headline comparison "
        "in the results stays YOLO11n @640 against Faster R-CNN @640; this table only "
        "answers what more would buy.",
        "",
        "| variant | mAP@0.5 | Δ | mAP@0.5:0.95 | mAP small | FP @0.5 | ms/img | speed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['mAP_0.5']} | {r['d_mAP_0.5']:+} | {r['mAP_0.5:0.95']} | "
            f"{r['mAP_small']} | {r['fp_at_0.50']} | {r['mean_ms']} | {r['speed_ratio']}x |"
        )
    lines += ["", "Δ and speed are relative to the baseline row.", ""]
    (RESULTS / "levers.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'levers.md'}")


if __name__ == "__main__":
    main()
