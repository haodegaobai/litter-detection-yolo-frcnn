"""Fetch the TACO images that the first pass could not get.

On 2026-10-28 the official CDN (img.tacodataset.org) was unreachable and Flickr
answered HTTP 429 at IP level, so only 828 of the 1500 images were downloaded and
every downstream number was computed on that subset. On 2026-11-10 the official
CDN came back: a probe of eight missing images returned 8/8 with HTTP 200 in
6 seconds.

This script fills the gap without touching anything already on disk:

* idempotent -- an image that exists and decodes is skipped;
* parallel -- 6 workers, ~0.75 s per image, so the whole set is minutes;
* non-destructive -- it only ever adds files under data/raw/images_orig.

    python src/download_missing.py            # fetch what is missing
    python src/download_missing.py --limit 5  # smoke test

It does NOT re-split, retrain or rewrite any result. After it finishes, the
pipeline is rebuilt explicitly by prepare_data.py and the training scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

from data_utils import RAW

ANN = RAW / "annotations.json"
DEST = RAW / "images_orig"
UA = "curl/8.9.1"  # a browser UA gets 429 from Flickr; this one does not


def target_for(file_name: str) -> Path:
    return DEST / file_name


def is_good(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 20000:
        return False
    im = cv2.imread(str(path))
    return im is not None and im.size > 0


def fetch(url: str, dest: Path, retries: int = 3) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        r = subprocess.run(
            ["curl", "-s", "-L", "--max-time", "40", "-A", UA, "-o", str(tmp), url],
            capture_output=True,
        )
        if r.returncode == 0 and is_good(tmp):
            tmp.replace(dest)
            return True, "ok"
    tmp.unlink(missing_ok=True)
    return False, "failed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw = json.loads(ANN.read_text(encoding="utf-8"))
    todo = []
    for im in raw["images"]:
        dest = target_for(im["file_name"])
        if not is_good(dest):
            todo.append((im["flickr_url"], dest, im["file_name"]))
    if args.limit:
        todo = todo[: args.limit]

    total = len(raw["images"])
    print(f"{total} images in annotations, {len(todo)} to fetch")
    if not todo:
        print("nothing to do")
        return

    ok = fail = 0
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, u, d): name for u, d, name in todo}
        for fut in as_completed(futures):
            name = futures[fut]
            good, _ = fut.result()
            done += 1
            if good:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED {name}", file=sys.stderr)
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}  ok={ok} fail={fail}", flush=True)

    have = sum(1 for im in raw["images"] if is_good(target_for(im["file_name"])))
    print(f"done: {ok} fetched, {fail} failed; {have}/{total} images now on disk")
    if have < total:
        print("re-run this script to retry the failures")


if __name__ == "__main__":
    main()
