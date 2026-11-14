"""Fallback downloader for images the first pass could not fetch.

The `_o` suffix in a Flickr URL means "original size". A fair number of the
TACO source photos have been deleted or made private since 2020, so the original
is simply gone. Flickr, however, keeps derived sizes for photos that still
exist, and the secret in the URL is shared across sizes, so `_b` (1024 px long
side) can often be substituted for `_o`.

Resolution note: we train at 640 and resize everything to a 1280 long side
anyway, so a 1024 px substitute still covers a 640 input completely. Images
recovered this way are logged in `data/raw/download_fallback_log.json` so the
README can state exactly how many were affected instead of silently mixing
resolutions.

Run after download_taco.py. Safe to re-run: anything already present is skipped.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data_utils import RAW

DEST = RAW / "images_orig"
LOG = RAW / "download_fallback_log.json"
MIN_BYTES = 2000
WORKERS = int(os.environ.get("TACO_WORKERS", "8"))
TIMEOUT = 90
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# tried in order; `_o` keeps the original, the rest are progressively smaller
SIZE_SUFFIXES = ["_o.png", "_o.jpg", "_b.jpg", "_c.jpg", "_z.jpg"]


def candidates(url: str) -> list[str]:
    if "staticflickr.com" not in url:
        return [url]
    stem = url.rsplit("/", 1)[-1]              # e.g. 33978196618_e30a59e0a8_o.png
    photo_id, secret, _ = stem.rsplit("_", 2)
    base = url.rsplit("/", 1)[0]
    out = []
    for suf in SIZE_SUFFIXES:
        cand = f"{base}/{photo_id}_{secret}{suf}"
        if cand not in out:
            out.append(cand)
    # the modern hostname serves the same objects
    out += [c.replace("farm", "live.static", 1) for c in list(out)]
    return out


def fetch(job: tuple[str, str]) -> tuple[str, str, str]:
    rel, url = job
    path = DEST / rel
    if path.exists() and path.stat().st_size >= MIN_BYTES:
        return rel, "already", ""
    path.parent.mkdir(parents=True, exist_ok=True)
    for cand in candidates(url):
        try:
            req = Request(cand, headers={"User-Agent": UA, "Referer": "https://www.flickr.com/"})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES:
                continue
            path.write_bytes(data)
            return rel, "recovered", cand
        except (HTTPError, URLError, OSError):
            time.sleep(0.4)
    return rel, "failed", url


def main() -> None:
    cur = json.loads((RAW / "annotations.json").read_text(encoding="utf-8"))
    old = json.loads((RAW / "annotations_flickr.json").read_text(encoding="utf-8"))
    url_by_id = {i["id"]: i.get("flickr_url", "") for i in old["images"]}
    jobs = [
        (img["file_name"], url_by_id[img["id"]])
        for img in cur["images"]
        if not (DEST / img["file_name"]).exists()
        or (DEST / img["file_name"]).stat().st_size < MIN_BYTES
    ]
    print(f"{len(jobs)} images still missing")
    if not jobs:
        return

    log: dict[str, str] = {}
    counts = {"recovered": 0, "failed": 0, "already": 0}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(fetch, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            rel, status, detail = f.result()
            counts[status] += 1
            if status == "recovered":
                log[rel] = detail
            if n % 50 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} {counts}", flush=True)

    LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"\n{counts}")
    print(f"recovered via a smaller Flickr size: {len(log)} (logged to {LOG.name})")


if __name__ == "__main__":
    main()
