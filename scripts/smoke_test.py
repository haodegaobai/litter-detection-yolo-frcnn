"""Load the committed weights, run one image, and check the headline metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import models as M


def check_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required file: {path.relative_to(ROOT)}")


def check_metrics() -> None:
    path = ROOT / "runs" / "results" / "eval_test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "YOLO11n": 0.1806,
        "FasterRCNN": 0.1615,
    }
    for model, value in expected.items():
        actual = data["models"][model]["map"]["mAP_0.5"]
        if abs(actual - value) > 1e-6:
            raise SystemExit(f"{model} mAP@0.5 mismatch: {actual} != {value}")
        print(f"metric ok: {model:<12} mAP@0.5 = {actual}")


def check_yolo() -> None:
    weights = ROOT / "runs" / "yolo" / "best.pt"
    model = M.load_yolo(weights)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    preds = M.predict_yolo(model, [frame])
    if len(preds) != 1:
        raise SystemExit(f"YOLO returned {len(preds)} predictions for one image")
    pred = preds[0]
    if pred["boxes"].ndim != 2 or pred["boxes"].shape[1] != 4:
        raise SystemExit(f"unexpected box shape: {tuple(pred['boxes'].shape)}")
    print(f"yolo ok: {weights.relative_to(ROOT)} -> {pred['boxes'].shape[0]} boxes")


def check_frcnn() -> None:
    weights = ROOT / "runs" / "frcnn" / "best.pt"
    if not weights.exists():
        print("frcnn skipped: runs/frcnn/best.pt not present (see Releases)")
        return
    model = M.load_frcnn(weights)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    preds = M.predict_frcnn(model, [frame], chunk=1)
    if len(preds) != 1:
        raise SystemExit(f"Faster R-CNN returned {len(preds)} predictions for one image")
    print(f"frcnn ok: {weights.relative_to(ROOT)} -> {preds[0]['boxes'].shape[0]} boxes")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frcnn", action="store_true", help="also load Faster R-CNN if the weight exists")
    args = ap.parse_args()

    for rel in [
        "runs/yolo/best.pt",
        "runs/yolo_scene/best.pt",
        "yolo11n.pt",
        "runs/results/eval_test.json",
    ]:
        check_file(ROOT / rel)

    check_metrics()
    check_yolo()
    if args.frcnn:
        check_frcnn()
    else:
        print("frcnn skipped: pass --frcnn to load it after downloading the Release weight")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
