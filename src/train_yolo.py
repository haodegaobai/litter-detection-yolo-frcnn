"""Task: fine-tune YOLO11n on the folded TACO dataset.

Training choices that matter for the comparison
-----------------------------------------------
* `imgsz=640` -- matches the input size Faster R-CNN is pinned to. Comparing a
  640 YOLO with an 800 R-CNN would be comparing input resolution, not
  architecture.
* `cache="ram"` -- 1500 resized JPEGs fit in memory, so the dataloader never
  becomes the bottleneck and the epoch time reflects GPU work.
* `deterministic=True` + fixed seed -- two runs should land on the same weights.
* `patience=20` -- TACO is small and long-tailed; early stopping is a guard
  against burning an hour on a model that has already plateaued.
* `plots=False` -- curves are drawn by plot_curves.py from the CSV, so the
  figures in this project come from one code path for both models.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO

from data_utils import ROOT, YOLO_ROOT

SEED = int(os.environ.get("YOLO_SEED", 42))
EPOCHS = int(os.environ.get("YOLO_EPOCHS", 100))
IMGSZ = int(os.environ.get("YOLO_IMGSZ", 640))
BATCH = int(os.environ.get("YOLO_BATCH", 16))
RUN_NAME = os.environ.get("YOLO_RUN_NAME", "yolo11n_litter")


def main() -> None:
    out_root = ROOT / "runs" / os.environ.get("YOLO_OUT", "yolo")
    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(os.environ.get("YOLO_WEIGHTS", "yolo11n.pt"))  # COCO-pretrained
    t0 = time.time()
    model.train(
        data=str(os.environ.get("YOLO_DATA") or (YOLO_ROOT / "data.yaml")),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        patience=20,
        cache="ram",
        workers=4,
        device=0,
        project=str(out_root),
        name=RUN_NAME,
        exist_ok=True,
        plots=False,
        # AMP off: ultralytics verifies AMP by downloading a model from GitHub,
        # which fails on this network and costs ~40 s of retries per run. FP32 is
        # also what benchmark.py times, so the two stay consistent.
        amp=False,
        verbose=True,
    )
    train_sec = round(time.time() - t0, 1)

    run_dir = out_root / RUN_NAME
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training finished but {best} is missing")

    # keep a stable path for the rest of the pipeline
    stable = out_root / "best.pt"
    shutil.copy2(best, stable)

    info = {
        "epochs_requested": EPOCHS,
        "train_sec": train_sec,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "seed": SEED,
        "weights": str(stable),
        "run_dir": str(run_dir),
        "dataset": json.loads((ROOT / "data" / "dataset_summary.json").read_text(encoding="utf-8"))["totals"],
    }
    (out_root / "train_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"done in {train_sec / 60:.1f} min -> {stable}")


if __name__ == "__main__":
    main()
