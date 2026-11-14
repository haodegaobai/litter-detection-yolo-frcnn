"""Keep retrying the image download until nothing is missing.

Flickr's CDN applies an IP-level rate limit that outlives any single run: once
it trips, every request returns 429 regardless of User-Agent, and it only clears
after a cooldown of unknown length. Rather than babysit that by hand, this wraps
`download_taco.py` in a loop: run, sleep, run again. Each pass is resumable, so a
pass that recovers 20 images before hitting the limit again still makes progress.

    python src/download_retry.py            # default: 4 min between passes, 40 passes
    python src/download_retry.py --sleep 120 --passes 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import download_taco
from data_utils import RAW

DEST = RAW / "images_orig"


def missing_count() -> int:
    cur = json.loads((RAW / "annotations.json").read_text(encoding="utf-8"))
    return sum(1 for i in cur["images"] if not (DEST / i["file_name"]).exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=int, default=240)
    ap.add_argument("--passes", type=int, default=40)
    args = ap.parse_args()

    for p in range(1, args.passes + 1):
        n = missing_count()
        print(f"\n=== pass {p}/{args.passes}: {n} images missing ===", flush=True)
        if n == 0:
            print("download complete")
            return
        try:
            download_taco.main()
        except Exception as exc:  # a network hiccup must not kill the loop
            print(f"pass {p} raised {type(exc).__name__}: {exc}", flush=True)
        if missing_count() == 0:
            print("download complete")
            return
        if p < args.passes:
            print(f"sleeping {args.sleep}s before the next pass", flush=True)
            time.sleep(args.sleep)

    print(f"gave up after {args.passes} passes, {missing_count()} still missing")


if __name__ == "__main__":
    main()
