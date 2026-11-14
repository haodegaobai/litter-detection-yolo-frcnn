"""Training curves for both models, from one plotting path.

YOLO's per-epoch numbers come from `runs/yolo/<run>/results.csv` (written by
ultralytics), Faster R-CNN's from `runs/frcnn/history.json` (written by
train_frcnn.py). Both are validation mAP@0.5 computed by *ultralytics'* validator
for YOLO and by torchmetrics for R-CNN -- so these curves are for diagnosing
training, NOT for the results table. The results table uses common_eval.py on
both models, as stated in the results.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import ROOT

FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_yolo_csv(path: Path) -> dict[str, list[float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = {k.strip(): k for k in rows[0]}
    out: dict[str, list[float]] = {}
    for name in ("epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                 "train/box_loss", "train/cls_loss", "train/dfl_loss"):
        if name in keys:
            out[name] = [float(r[keys[name]]) for r in rows]
    return out


def main() -> None:
    yolo_csv = ROOT / "runs" / "yolo" / "yolo11n_litter" / "results.csv"
    frcnn_json = ROOT / "runs" / "frcnn" / "history.json"
    if not yolo_csv.exists() or not frcnn_json.exists():
        raise SystemExit("train both models first")

    y = read_yolo_csv(yolo_csv)
    f = json.loads(frcnn_json.read_text(encoding="utf-8"))
    f_ep = [r["epoch"] for r in f]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    ax = axes[0]
    ax.plot(y["epoch"], y["metrics/mAP50(B)"], label="YOLO11n val mAP@0.5", lw=1.8)
    ax.plot(y["epoch"], y["metrics/mAP50-95(B)"], label="YOLO11n val mAP@0.5:0.95", lw=1.4, ls="--")
    ax.plot(f_ep, [r["val_mAP_0.5"] for r in f], label="Faster R-CNN val mAP@0.5", lw=1.8)
    ax.plot(f_ep, [r["val_mAP_0.5:0.95"] for r in f],
            label="Faster R-CNN val mAP@0.5:0.95", lw=1.4, ls="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP (own validators)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    ax.plot(f_ep, [r["train_loss"] for r in f], label="Faster R-CNN train loss", lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Faster R-CNN training loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIG / "fig_curves.png"
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")

    # overfitting evidence, printed rather than asserted
    best = max(f, key=lambda r: r["val_mAP_0.5"])
    print(
        f"R-CNN best val mAP@0.5 {best['val_mAP_0.5']} at epoch {best['epoch']}; "
        f"final train loss {f[-1]['train_loss']} (first {f[0]['train_loss']})"
    )
    ybest = max(range(len(y["metrics/mAP50(B)"])), key=lambda i: y["metrics/mAP50(B)"][i])
    print(
        f"YOLO best val mAP@0.5 {y['metrics/mAP50(B)'][ybest]} at epoch "
        f"{int(y['epoch'][ybest])} of {int(y['epoch'][-1])}"
    )


if __name__ == "__main__":
    main()
