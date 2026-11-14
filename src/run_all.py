"""Run the whole pipeline in order, skipping stages whose output already exists.

Convenience wrapper only -- each stage is still a standalone script, and running
them one at a time is the intended workflow. This exists so a fresh clone can be
brought up with one command and the artefacts are produced in a fixed order.

    python src/run_all.py            # skip stages that are already done
    python src/run_all.py --force    # redo everything
    python src/run_all.py --from 4   # start at stage 4
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from data_utils import ROOT

STAGES = [
    ("download images", "download_taco.py", ROOT / "data" / "raw" / "images_orig"),
    ("prepare dataset", "prepare_data.py", ROOT / "data" / "dataset_summary.json"),
    ("eda figures", "eda.py", ROOT / "figures" / "fig1_class_distribution.png"),
    ("train YOLO", "train_yolo.py", ROOT / "runs" / "yolo" / "best.pt"),
    ("train Faster R-CNN", "train_frcnn.py", ROOT / "runs" / "frcnn" / "best.pt"),
    ("training curves", "plot_curves.py", ROOT / "figures" / "fig_curves.png"),
    ("evaluate test split", "run_eval.py", ROOT / "runs" / "results" / "eval_test.json"),
    ("speed benchmark", "benchmark.py", ROOT / "runs" / "results" / "speed_benchmark.json"),
    ("error analysis", "error_analysis.py", ROOT / "runs" / "results" / "error_stats.json"),
    ("demo configuration cost", "deploy_config.py", ROOT / "runs" / "results" / "deploy_config.json"),
]


def done(sentinel) -> bool:
    if sentinel.is_dir():
        return any(sentinel.iterdir())
    return sentinel.exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1)
    args = ap.parse_args()

    for i, (label, script, sentinel) in enumerate(STAGES, 1):
        if i < args.start:
            continue
        if not args.force and done(sentinel):
            print(f"[{i}/{len(STAGES)}] {label:<22} skip (output exists)")
            continue
        print(f"\n[{i}/{len(STAGES)}] {label}  ->  {script}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, str(ROOT / "src" / script)], cwd=str(ROOT))
        if rc != 0:
            print(f"FAILED ({rc}) after {time.time() - t0:.0f}s: {script}")
            return rc
        print(f"    ok in {time.time() - t0:.0f}s")
    print("\nall stages complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
