"""Speed benchmark under one protocol for both models.

Protocol (identical for both)
-----------------------------
* batch size 1, FP32
* 640 px input
* images pre-decoded into memory, so the number is model cost, not disk cost
* 10 warm-up iterations discarded (CUDA kernel autotuning, cudnn workspace)
* 100 timed iterations
* torch.cuda.synchronize() around the timed region, otherwise the GPU queue is
  measured instead of the GPU work

Reported: mean / p50 / p95 latency, FPS from the mean, parameter count, and
weights size on disk.

Note on what this does and does not measure: this is *model* throughput, not the
camera loop. The live demo's on-screen FPS is capped by the 30 fps webcam and by
frame capture, so the two numbers must not be conflated in the results.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

from data_utils import ROOT, load_gt
from models import DEVICE, load_frcnn, load_yolo, predict_frcnn, predict_yolo, read_bgr

WARMUP = 10
ITERS = 100
RESULTS = ROOT / "runs" / "results"


def timed(fn, frames: list) -> dict:
    for i in range(WARMUP):
        fn(frames[i % len(frames)])
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    lat = []
    for i in range(ITERS):
        t0 = time.perf_counter()
        fn(frames[i % len(frames)])
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)

    lat_sorted = sorted(lat)
    mean = statistics.fmean(lat)
    return {
        "mean_ms": round(mean, 2),
        "p50_ms": round(lat_sorted[len(lat_sorted) // 2], 2),
        "p95_ms": round(lat_sorted[int(len(lat_sorted) * 0.95)], 2),
        "min_ms": round(min(lat), 2),
        "max_ms": round(max(lat), 2),
        "fps": round(1000.0 / mean, 1),
        "n_iters": ITERS,
    }


def n_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    records = load_gt("test")
    frames = [read_bgr(r["path"]) for r in records]
    print(f"{len(frames)} frames preloaded, device={DEVICE}")

    out = {"device": str(DEVICE), "imgsz": 640, "warmup": WARMUP, "iters": ITERS, "models": {}}

    yolo = load_yolo(ROOT / "runs" / "yolo" / "best.pt")
    print("YOLO ...", flush=True)
    out["models"]["YOLO11n"] = timed(lambda f: predict_yolo(yolo, [f]), frames)
    out["models"]["YOLO11n"]["params"] = n_params(yolo.model)
    out["models"]["YOLO11n"]["weights_mb"] = round(
        (ROOT / "runs" / "yolo" / "best.pt").stat().st_size / 1e6, 2
    )

    frcnn = load_frcnn(ROOT / "runs" / "frcnn" / "best.pt")
    print("Faster R-CNN ...", flush=True)
    out["models"]["FasterRCNN"] = timed(lambda f: predict_frcnn(frcnn, [f]), frames)
    out["models"]["FasterRCNN"]["params"] = n_params(frcnn)
    out["models"]["FasterRCNN"]["weights_mb"] = round(
        (ROOT / "runs" / "frcnn" / "best.pt").stat().st_size / 1e6, 2
    )

    (RESULTS / "speed_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# Speed benchmark",
        "",
        f"Device: {out['device']} · {ITERS} timed iterations, {WARMUP} warm-up discarded · batch 1 · FP32 · 640 px",
        "",
        "| | YOLO11n | Faster R-CNN |",
        "|---|---|---|",
    ]
    y, f = out["models"]["YOLO11n"], out["models"]["FasterRCNN"]
    for k, label in [
        ("mean_ms", "mean ms / image"),
        ("p50_ms", "p50 ms"),
        ("p95_ms", "p95 ms"),
        ("fps", "FPS"),
        ("params", "parameters"),
        ("weights_mb", "weights (MB)"),
    ]:
        lines.append(f"| {label} | {y[k]} | {f[k]} |")
    lines.append(f"| speed-up | {round(f['mean_ms'] / y['mean_ms'], 2)}x | 1.00x |")
    (RESULTS / "speed_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
