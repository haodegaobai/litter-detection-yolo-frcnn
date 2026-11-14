"""Build the prepared dataset: fold 60 TACO classes to 6, split, resize, export.

One source of truth
-------------------
Everything downstream (YOLO and Faster R-CNN alike) reads the COCO files written
here. The split is created exactly once, deterministically (seed 42), so the two
detectors cannot accidentally see different data.

Why a split has to be invented
------------------------------
TACO is published as a single pool of 1500 images with no official train/val/test
partition (unlike Roboflow exports, which ship one). A stratified image-level
split at 70/15/15 is therefore created here, stratified by each image's *primary*
superclass (the superclass with the most instances in that image) so that every
class appears in every split.

Known limitation, stated rather than hidden: TACO images were collected in
batches by different photographers, and some near-duplicate shots of the same
scene exist. An image-level random split can therefore leak a scene across
splits and inflate test mAP. The size of that effect is not measured here.

Why images are resized to a 1280 long side
------------------------------------------
Median long side in TACO is 3264 px and the largest is 6000 px. Training runs at
640 px, so decoding a 3264 px JPEG every epoch is pure waste -- the letterbox
step would throw the extra pixels away anyway. Resizing to 1280 keeps every pixel
that a 640 (or even a 960) input can use, cuts decode time by roughly 3x, and
makes the whole dataset fit comfortably on disk. Boxes are rescaled with the
images; no annotation is dropped by the resize.
"""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from data_utils import CLASSES, NUM_CLASSES, RAW, ROOT, SUPERCLASS_MAP, YOLO_ROOT, COCO_ROOT

MAX_SIDE = 1280
JPEG_QUALITY = 92
SEED = 42
WORKERS = int(os.environ.get("PREP_WORKERS", "8"))
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / valid / test

src_of_global: dict[int, Path] = {}


def load_raw() -> dict:
    return json.loads((RAW / "annotations.json").read_text(encoding="utf-8"))


def superclass_index(raw: dict) -> dict[int, int]:
    """TACO category_id -> 0-based superclass index."""
    missing = [c["name"] for c in raw["categories"] if c["name"] not in SUPERCLASS_MAP]
    if missing:
        raise RuntimeError(f"unmapped TACO categories: {missing}")
    return {c["id"]: CLASSES.index(SUPERCLASS_MAP[c["name"]]) for c in raw["categories"]}


def collect(raw: dict, cat_map: dict[int, int]) -> dict[int, list[tuple]]:
    """image_id -> list of (x0, y0, x1, y1, superclass_idx), clipped to the frame."""
    sizes = {i["id"]: (float(i["width"]), float(i["height"])) for i in raw["images"]}
    out: dict[int, list[tuple]] = defaultdict(list)
    clipped = dropped = 0
    for ann in raw["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        W, H = sizes[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(W, x + bw), min(H, y + bh)
        if x1 <= x0 or y1 <= y0:
            dropped += 1
            continue
        if (x0, y0, x1, y1) != (x, y, x + bw, y + bh):
            clipped += 1
        out[ann["image_id"]].append((x0, y0, x1, y1, cat_map[ann["category_id"]]))
    print(f"annotations: clipped to frame={clipped}, dropped (degenerate)={dropped}")
    return out


def primary_class(boxes: list[tuple]) -> int:
    return Counter(b[4] for b in boxes).most_common(1)[0][0]


def make_split(images: list[dict], per_image: dict[int, list[tuple]]) -> dict[str, list[dict]]:
    """Stratified by primary superclass, deterministic, every class in every split."""
    by_class: dict[int, list[dict]] = defaultdict(list)
    for img in images:
        by_class[primary_class(per_image[img["id"]])].append(img)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda i: i["id"])
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(SPLIT_FRACTIONS[1] * n)) if n >= 3 else 0
        n_test = max(1, round(SPLIT_FRACTIONS[2] * n)) if n >= 3 else 0
        if n - n_val - n_test < 1:
            n_val = n_test = 0
        splits["test"] += group[:n_test]
        splits["valid"] += group[n_test:n_test + n_val]
        splits["train"] += group[n_test + n_val:]

    for k in splits:
        splits[k].sort(key=lambda i: i["id"])
    return splits


def resize_and_save(src: Path, dst: Path) -> tuple[int, int, float]:
    """Save a resized JPEG; return (width, height, scale) of the *output*."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        if scale < 1.0:
            im = im.resize((nw, nh), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return nw, nh, scale


def process_image(args: tuple) -> tuple[dict, list[dict], dict]:
    """Resize one image and rescale its boxes. Runs in a thread pool.

    Decoding a 3000x4000 JPEG and re-encoding it is almost entirely C code that
    releases the GIL, so threads (not processes) give close to linear speed-up
    here without the cost of pickling tensors between processes.
    """
    img, boxes, split = args
    new_name = f"{img['file_name'].replace('/', '_').rsplit('.', 1)[0]}.jpg"
    dst = COCO_ROOT / split / new_name
    nw, nh, scale = resize_and_save(src_of_global[img["id"]], dst)

    anns = []
    for x0, y0, x1, y1, cls in boxes:
        bw, bh = (x1 - x0) * scale, (y1 - y0) * scale
        if bw <= 1 or bh <= 1:
            continue
        anns.append(
            {
                "image_id": img["id"],
                "category_id": cls,
                "bbox": [round(x0 * scale, 2), round(y0 * scale, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
            }
        )
    return {"id": img["id"], "file_name": new_name, "width": nw, "height": nh}, anns, {
        "scale": scale
    }


def main() -> None:
    global src_of_global
    raw = load_raw()
    cat_map = superclass_index(raw)
    per_image = collect(raw, cat_map)

    kept = [i for i in raw["images"] if per_image.get(i["id"])]
    print(f"images: {len(raw['images'])} total, {len(kept)} with >=1 mapped annotation")

    splits = make_split(kept, per_image)

    if COCO_ROOT.exists():
        shutil.rmtree(COCO_ROOT)
    if YOLO_ROOT.exists():
        shutil.rmtree(YOLO_ROOT)

    src_of_global = {i["id"]: RAW / "images_orig" / i["file_name"] for i in raw["images"]}
    missing = {iid for iid, p in src_of_global.items() if not p.exists()}
    if missing:
        # Some of TACO's source photos have been deleted from Flickr since 2020
        # and cannot be recovered at any size. They are dropped and counted here
        # rather than silently skipped, so the README can state the real size of
        # the dataset instead of the nominal 1500.
        print(f"WARNING: {len(missing)} of {len(raw['images'])} images unavailable; dropping them")
        for split in splits:
            splits[split] = [i for i in splits[split] if i["id"] not in missing]
        kept_n = sum(len(v) for v in splits.values())
        print(f"         {kept_n} images remain across all splits")

    summary = {"class_names": CLASSES, "splits": {}, "taxonomy": {}}
    class_totals = Counter()

    for split, imgs in splits.items():
        img_dir = COCO_ROOT / split
        img_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(img, per_image[img["id"]], split) for img in imgs]

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            results = list(pool.map(process_image, jobs))

        coco = {
            "info": {"description": f"TACO folded to {NUM_CLASSES} superclasses ({split})"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(CLASSES)],
        }
        ann_id = 1
        per_class = Counter()
        for image_entry, anns, _ in results:
            coco["images"].append(image_entry)
            for a in anns:
                a["id"] = ann_id
                ann_id += 1
                coco["annotations"].append(a)
                per_class[CLASSES[a["category_id"]]] += 1
                class_totals[CLASSES[a["category_id"]]] += 1

        coco["images"].sort(key=lambda i: i["id"])
        coco["annotations"].sort(key=lambda a: a["id"])
        (img_dir / "_annotations.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False), encoding="utf-8"
        )
        summary["splits"][split] = {
            "images": len(coco["images"]),
            "objects": len(coco["annotations"]),
            "per_class": {k: per_class.get(k, 0) for k in CLASSES},
        }
        print(f"{split:>6}: {len(coco['images']):>4} images, {len(coco['annotations']):>5} objects")

    # ---- YOLO export: hardlink the images, write normalised labels ----
    for split in splits:
        img_out = YOLO_ROOT / "images" / split
        lbl_out = YOLO_ROOT / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        coco = json.loads((COCO_ROOT / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        by_img = defaultdict(list)
        for ann in coco["annotations"]:
            by_img[ann["image_id"]].append(ann)
        for im in coco["images"]:
            src = COCO_ROOT / split / im["file_name"]
            dst = img_out / im["file_name"]
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            W, H = float(im["width"]), float(im["height"])
            lines = []
            for ann in by_img.get(im["id"], []):
                x, y, w, h = ann["bbox"]
                lines.append(
                    f"{ann['category_id']} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} "
                    f"{w / W:.6f} {h / H:.6f}"
                )
            (lbl_out / f"{Path(im['file_name']).stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    (YOLO_ROOT / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {YOLO_ROOT.as_posix()}",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                "",
                f"nc: {NUM_CLASSES}",
                f"names: {CLASSES}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    summary["taxonomy"] = {
        "source_categories": len(raw["categories"]),
        "superclasses": {k: class_totals[k] for k in CLASSES},
        "mapping": SUPERCLASS_MAP,
    }
    summary["totals"] = {
        "images": sum(s["images"] for s in summary["splits"].values()),
        "objects": sum(s["objects"] for s in summary["splits"].values()),
    }
    (ROOT / "data" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nsuperclass instance totals:", dict(class_totals))
    print("totals:", summary["totals"])


if __name__ == "__main__":
    main()
