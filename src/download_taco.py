"""Download the TACO images.

Why this script exists
----------------------
`img.tacodataset.org` (the current image host) is unreachable from this network,
and so is `universe.roboflow.com`. The original Flickr URLs *are* reachable, and
they are still recorded in the parent of commit `3844bcd`
("Serve images from Cloudflare R2 instead of Flickr") in `pedropro/TACO`.
That commit's `data/annotations.json` is saved here as `annotations_flickr.json`.

The two annotation files are byte-for-byte equivalent in *content* (same 1500
images, same 4784 annotations, same 60 categories, same ids and file names);
only the `flickr_url` field differs. So the Flickr URLs are joined onto the
current `annotations.json` by image id -- no other field is taken from the old
file.

Resumable: files already on disk with a plausible size are skipped.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEST = RAW / "images_orig"
FAIL = RAW / "download_failures.txt"

WORKERS = int(os.environ.get("TACO_WORKERS", "12"))
RETRIES = 3
TIMEOUT = 90
MIN_BYTES = 2000

UA = "curl/8.9.1"
# Measured on this network: Flickr's CDN answers a *browser* User-Agent with
# HTTP 429 (Too Many Requests) almost immediately, while a plain curl-style UA
# or no UA at all gets HTTP 200 for the same object. Verified by holding the URL
# fixed and varying only the header. Hence the non-browser UA below.
#
# The CDN also enforces an IP-level rate limit that is independent of the UA:
# once it trips, every request 429s regardless of header. So the downloader is
# throttled to a fixed spacing between request *starts* rather than relying on
# worker count, and a 429 triggers a long cooldown instead of a quick retry.
RETRY_429_SLEEP = float(os.environ.get("TACO_429_SLEEP", "30"))
MIN_INTERVAL = float(os.environ.get("TACO_INTERVAL", "1.0"))

_rate_lock = threading.Lock()
_last_start = [0.0]


def _throttle() -> None:
    """Space request starts by at least MIN_INTERVAL seconds, across all threads."""
    with _rate_lock:
        now = time.monotonic()
        wait = _last_start[0] + MIN_INTERVAL - now
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_start[0] = now


def build_url_map() -> dict[int, tuple[str, str]]:
    cur = json.loads((RAW / "annotations.json").read_text(encoding="utf-8"))
    old = json.loads((RAW / "annotations_flickr.json").read_text(encoding="utf-8"))
    url_by_id = {i["id"]: i.get("flickr_url", "") for i in old["images"]}
    out = {}
    for img in cur["images"]:
        url = url_by_id.get(img["id"], "")
        if not url:
            raise RuntimeError(f"no source url for image id {img['id']}")
        out[img["id"]] = (img["file_name"], url)
    return out


def fetch(job: tuple[int, str, str]) -> tuple[int, str, int, str]:
    image_id, rel, url = job
    path = DEST / rel
    if path.exists() and path.stat().st_size >= MIN_BYTES:
        return image_id, rel, path.stat().st_size, "skip"

    path.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(1, RETRIES + 1):
        _throttle()
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            if len(data) < MIN_BYTES:
                raise ValueError(f"too small ({len(data)} bytes)")
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            return image_id, rel, len(data), "ok"
        except HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code == 429:
                # Once the IP-level limit trips, every subsequent request 429s for
                # a while. Retrying in-place just burns minutes per image, so bail
                # out immediately and let download_retry.py sleep and come back.
                return image_id, rel, 0, "FAIL HTTP 429 (rate limited)"
            time.sleep(1.5 * attempt)
        except (URLError, ValueError, OSError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * attempt)
    return image_id, rel, 0, f"FAIL {last}"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    jobs = [(iid, rel, url) for iid, (rel, url) in build_url_map().items()]
    print(f"{len(jobs)} images to fetch, {WORKERS} workers", flush=True)

    ok = skipped = 0
    failures: list[str] = []
    total_bytes = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            image_id, rel, size, status = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += size
            elif status == "skip":
                skipped += 1
            else:
                failures.append(f"{image_id}\t{rel}\t{status}")
            if n % 25 == 0 or n == len(jobs):
                el = time.time() - t0
                rate = n / el if el else 0
                eta = (len(jobs) - n) / rate if rate else 0
                print(
                    f"  {n:>5}/{len(jobs)}  ok={ok} skip={skipped} fail={len(failures)}  "
                    f"{total_bytes / 1e9:.2f} GB  {rate:.1f} img/s  eta {eta / 60:.1f} min",
                    flush=True,
                )

    FAIL.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    el = time.time() - t0
    print(f"\ndone in {el / 60:.1f} min: ok={ok} skipped={skipped} failed={len(failures)}")
    print(f"downloaded {total_bytes / 1e9:.2f} GB into {DEST}")
    if failures:
        print(f"failures logged to {FAIL}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
