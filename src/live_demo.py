"""Real-time litter detection from a webcam (or a video file).

This is the deliverable that has to survive a live demo, so it is written to
fail loudly and degrade gracefully:

* `cv2.CAP_DSHOW` on Windows -- without it, opening camera 0 can stall for
  several seconds while the MSMF backend enumerates devices.
* `--source` accepts a file, so the same code path can replay a recorded clip.
  That is the fallback if the camera or the room lighting misbehaves on the day.
* `--no-display` writes an annotated MP4 instead of opening a window, which is
  also how the pipeline gets tested on a machine without a display.
* `r` toggles recording of the *raw* frames (no boxes burned in), so the clip
  stays usable as evaluation footage for the offline Faster R-CNN pass.
* `s` writes a snapshot of the annotated frame.

Two frame rates, and they are not the same number
-------------------------------------------------
`loop` is wall-clock frames per second for the whole pipeline: capture,
preprocess, inference, NMS, draw. On a 30 fps webcam it cannot exceed 30.
`model` is 1000 / mean inference latency, i.e. what the GPU alone costs.

The HUD shows both. An earlier version printed only the second one under the
label "FPS", which made the demo look 3x faster than the pipeline actually was
and contradicted the earlier version. Both are now on screen, labelled.

Demo-safe mode (`--demo-safe`)
------------------------------
The model was trained on outdoor litter photographs only. Pointed at a
classroom it will happily label blackboard strokes as `cigarette`, because the
training set contains no image that says "this is not litter" and the cigarette
head is effectively untrained (AP@0.5:0.95 = 0.0039). That is a data-composition
problem, not something a confidence slider fixes.

`--demo-safe` therefore constrains the *demo surface* rather than pretending the
model improved:

    conf 0.60                 precision 0.6429 on the 3-class test set vs 0.5000 at 0.25
    3 classes only            cigarette and glass are dropped; their AP is ~0.004 and ~0.000
    aspect ratio <= 4         text strokes and rails are elongated; real litter is not
    min area 0.05% of frame   a single glyph is smaller than any real object
    stability 2 frames        a box must survive two consecutive frames to be drawn

Measured on a 450-frame litter clip: those four filters keep 0.71 boxes per frame
against 0.81 with confidence and classes alone, so they cost about 12% of the
detections and remove the flicker.

There is a fifth switch, `--roi-bottom`, that is **not** part of the preset on
purpose. Masking the top of the frame is the strongest defence when the camera
faces a board or a screen, but on a ground-level shot the litter sits in the
upper half too: measured on the same clip, adding `--roi-bottom 0.55` drops the
output from 0.71 to 0.18 boxes per frame. Use it when you know what the camera
is looking at, not as a default.

Each of those is disclosed on screen ("SAFE" badge) and in the README, so the
numbers quoted from the test split remain the numbers quoted from the test split.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from data_utils import CLASSES, ROOT
from models import DEVICE, load_frcnn, load_yolo, predict_frcnn, predict_yolo

PALETTE = [
    (66, 135, 245),    # plastic      - blue
    (245, 176, 66),    # paper        - amber
    (140, 140, 150),   # metal        - grey
    (80, 200, 200),    # glass        - cyan
    (90, 90, 235),     # cigarette    - red-ish
    (150, 200, 90),    # other        - green
]

# Classes that actually have a working head on this dataset. See the module
# docstring: cigarette 0.0039, glass 0.0002-0.0079 AP@0.5:0.95.
SAFE_CLASSES = [0, 1, 2]  # plastic, paper_cardboard, metal


# --------------------------------------------------------------------------- #
# post-processing
# --------------------------------------------------------------------------- #
def filter_pred(
    pred: dict,
    conf: float,
    classes: list[int] | None,
    max_aspect: float,
    min_area_frac: float,
    frame_area: float,
    roi_y0: int = 0,
) -> dict:
    """Apply the demo-time filters and return a new prediction dict.

    The ROI is enforced by dropping boxes whose *centre* sits above `roi_y0`,
    not by cropping the input. Cropping would shrink the crop to 640 px on its
    long side and make every object smaller, which is the opposite of what a
    demo needs; filtering keeps the model's view of the frame identical to
    evaluation while still guaranteeing nothing above the line is drawn.
    """
    boxes = pred["boxes"].numpy()
    scores = pred["scores"].numpy()
    labels = pred["labels"].numpy()

    keep = scores >= conf
    if classes is not None:
        keep &= np.isin(labels, classes)
    if len(boxes) and max_aspect > 0:
        w = np.clip(boxes[:, 2] - boxes[:, 0], 1e-6, None)
        h = np.clip(boxes[:, 3] - boxes[:, 1], 1e-6, None)
        aspect = np.maximum(w / h, h / w)
        keep &= aspect <= max_aspect
    if len(boxes) and min_area_frac > 0:
        area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
            boxes[:, 3] - boxes[:, 1], 0, None
        )
        keep &= area >= min_area_frac * frame_area
    if len(boxes) and roi_y0 > 0:
        centres = (boxes[:, 1] + boxes[:, 3]) / 2.0
        keep &= centres >= roi_y0

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-6)


class Stabiliser:
    """Keep a box only after it has appeared in `need` consecutive frames.

    Text on a blackboard flickers as the camera moves; a real object on a table
    does not. Matching is greedy by class + IoU, which is enough for a demo and
    avoids pulling in a tracker.
    """

    def __init__(self, need: int, iou_thr: float = 0.3):
        self.need = max(1, need)
        self.iou_thr = iou_thr
        self.prev: list[tuple[int, np.ndarray, int]] = []  # (label, box, hits)

    def __call__(self, pred: dict) -> dict:
        if self.need == 1:
            return pred

        boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]
        hits = [1] * len(boxes)
        used = [False] * len(self.prev)
        for i in range(len(boxes)):
            best, best_iou = -1, self.iou_thr
            for j, (plab, pbox, _) in enumerate(self.prev):
                if used[j] or plab != int(labels[i]):
                    continue
                v = _iou(boxes[i], pbox)
                if v >= best_iou:
                    best, best_iou = j, v
            if best >= 0:
                hits[i] = self.prev[best][2] + 1
                used[best] = True

        self.prev = [(int(labels[i]), boxes[i], hits[i]) for i in range(len(boxes))]
        keep = np.array([h >= self.need for h in hits], dtype=bool)
        if keep.size == 0:
            keep = np.zeros((0,), dtype=bool)
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}


# --------------------------------------------------------------------------- #
# person veto
# --------------------------------------------------------------------------- #
class PersonVeto:
    """Drop detections that are really a person.

    The detector was trained on litter crops only. It has never seen a human, so
    a face or a torso is just another textured region -- and with the class list
    pruned to three, whatever fires has to be called plastic, paper or metal.
    On a recording of the author holding a headset, eight detections survived the
    demo filters: three on the headset, three on a tissue, and two on his face.

    No geometric filter separates a face from a bottle: both are small, roughly
    square, and high-contrast. So a second COCO model -- which does know what a
    person is -- runs alongside and vetoes two cases:

      * the detection's centre falls inside the head region of a person box --
        the top 45% vertically and the middle 60% horizontally, which is where a
        face sits and a held object usually does not, or
      * the detection is large (> 10% of the frame) and overlaps a person box,
        which is what "the whole person read as plastic" looks like.

    A bottle held in front of the body is small and outside the head region, so
    it survives. This is a demo-time guard, not a fix: the fix is negative data.
    """

    HEAD_FRACTION = 0.45
    HEAD_WIDTH = 0.60
    BIG_AREA = 0.10
    PERSON_IOU = 0.30

    def __init__(self, weights, conf: float = 0.40, imgsz: int = 320):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def persons(self, frame) -> np.ndarray:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                               classes=[0], verbose=False, device=0 if DEVICE.type == "cuda" else "cpu")
        return r[0].boxes.xyxy.detach().cpu().numpy() if r and len(r[0].boxes) else np.zeros((0, 4))

    def apply(self, pred: dict, persons: np.ndarray, frame_area: float) -> tuple[dict, int]:
        if len(persons) == 0 or len(pred["scores"]) == 0:
            return pred, 0
        boxes = pred["boxes"]
        keep = np.ones(len(boxes), dtype=bool)
        for i, b in enumerate(boxes):
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            area = (b[2] - b[0]) * (b[3] - b[1]) / max(frame_area, 1.0)
            for p in persons:
                head_bottom = p[1] + (p[3] - p[1]) * self.HEAD_FRACTION
                margin = (p[2] - p[0]) * (1 - self.HEAD_WIDTH) / 2.0
                if (p[0] + margin <= cx <= p[2] - margin
                        and p[1] <= cy <= head_bottom):
                    keep[i] = False
                    break
                if area > self.BIG_AREA and _iou(b, p) > self.PERSON_IOU:
                    keep[i] = False
                    break
        return ({"boxes": boxes[keep], "scores": pred["scores"][keep],
                 "labels": pred["labels"][keep]}, int((~keep).sum()))



# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, pred: dict, roi_y0: int = 0) -> np.ndarray:
    """Draw boxes. `roi_y0` is where the ROI starts, so we can shade above it."""
    if roi_y0 > 0:
        shade = frame.copy()
        cv2.rectangle(shade, (0, 0), (frame.shape[1], roi_y0), (32, 32, 32), -1)
        cv2.addWeighted(shade, 0.45, frame, 0.55, 0, frame)
        cv2.line(frame, (0, roi_y0), (frame.shape[1], roi_y0), (90, 200, 240), 1)
        cv2.putText(frame, "outside detection area", (12, max(18, roi_y0 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1, cv2.LINE_AA)

    for (x1, y1, x2, y2), lab, sc in zip(pred["boxes"], pred["labels"], pred["scores"]):
        color = PALETTE[int(lab) % len(PALETTE)]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{CLASSES[int(lab)]} {sc:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Keep the label band on-canvas. A box touching the top edge would otherwise
        # draw its class name above the frame, leaving a bare rectangle on screen.
        ly = y1 if y1 - th - 6 >= 0 else min(y2, frame.shape[0] - 1)
        cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 4, ly), color, -1)
        cv2.putText(frame, text, (x1 + 2, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def hud(frame: np.ndarray, stats: dict, model_name: str) -> None:
    lines = [
        f"{model_name}  loop {stats['loop_fps']:.1f} FPS   model {stats['model_fps']:.1f} FPS",
        f"inference {stats['ms']:.1f} ms   p50 {stats['p50']:.1f}   p95 {stats['p95']:.1f}",
        f"detections: {stats['n_det']}   conf >= {stats['conf']:.2f}"
        + (f"   vetoed: {stats['vetoed']}" if stats.get("vetoed") else ""),
    ]
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, t, (12, 26 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)

    tag_y = 26 + 3 * 22
    if stats["safe"]:
        cv2.putText(frame, "SAFE MODE", (12, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "SAFE MODE", (12, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 200, 90), 1, cv2.LINE_AA)
    if stats["recording"]:
        cv2.putText(frame, "REC", (frame.shape[1] - 70, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "REC", (frame.shape[1] - 70, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (60, 60, 230), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index or path to a video file")
    ap.add_argument("--model", default="yolo", choices=["yolo", "frcnn"])
    ap.add_argument("--weights", default=None,
                    help="override the checkpoint, e.g. runs/yolo_scene/best.pt")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence floor (default 0.30, or 0.60 with --demo-safe)")
    ap.add_argument("--demo-safe", action="store_true",
                    help="conservative demo configuration; see module docstring")
    ap.add_argument("--classes", default=None,
                    help="comma-separated class indices to keep, e.g. 0,1,2")
    ap.add_argument("--roi-bottom", type=float, default=None,
                    help="fraction of frame height to detect in, measured from the "
                         "bottom (0.55 = bottom 55%%); 1.0 disables the mask")
    ap.add_argument("--max-aspect", type=float, default=None,
                    help="drop boxes whose width/height ratio exceeds this")
    ap.add_argument("--min-area", type=float, default=None,
                    help="drop boxes smaller than this fraction of the frame")
    ap.add_argument("--stable", type=int, default=None,
                    help="draw a box only after N consecutive frames")
    ap.add_argument("--person-veto", action="store_true",
                    help="use a COCO person detector to drop detections that are really a "
                         "face or a torso; on by default with --demo-safe")
    ap.add_argument("--no-person-veto", action="store_true",
                    help="disable the veto even in demo-safe mode")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default=None, help="annotated output video (with --no-display)")
    ap.add_argument("--record-dir", default=str(ROOT / "demo" / "raw"))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    # Resolve the configuration: explicit flags win, --demo-safe supplies defaults.
    conf = args.conf if args.conf is not None else (0.60 if args.demo_safe else 0.30)
    if args.classes is not None:
        classes = [int(c) for c in args.classes.split(",") if c.strip() != ""]
    else:
        classes = SAFE_CLASSES if args.demo_safe else None
    roi_bottom = args.roi_bottom if args.roi_bottom is not None else 1.0
    max_aspect = args.max_aspect if args.max_aspect is not None else (4.0 if args.demo_safe else 0.0)
    min_area = args.min_area if args.min_area is not None else (0.0005 if args.demo_safe else 0.0)
    stable = args.stable if args.stable is not None else (2 if args.demo_safe else 1)
    person_veto = (args.person_veto or args.demo_safe) and not args.no_person_veto

    weights = (Path(args.weights) if args.weights
               else ROOT / "runs" / ("yolo" if args.model == "yolo" else "frcnn") / "best.pt")
    if not weights.exists():
        raise SystemExit(f"missing weights: {weights}")

    src: str | int = int(args.source) if args.source.isdigit() else args.source
    if isinstance(src, int):
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on a live camera
    else:
        if not Path(src).exists():
            raise SystemExit(f"no such file: {src}")
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {args.source}")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("source opened but returned no frame")
    H, W = frame.shape[:2]
    frame_area = float(H * W)
    roi_y0 = 0 if roi_bottom >= 1.0 else int(H * (1.0 - roi_bottom))
    print(
        f"source={args.source} frame={W}x{H} device={DEVICE} "
        f"conf={conf:.2f} classes={classes or 'all'} roi_y0={roi_y0} "
        f"max_aspect={max_aspect or 'off'} stable={stable}"
    )

    model = load_yolo(weights) if args.model == "yolo" else load_frcnn(weights)
    stabiliser = Stabiliser(stable)
    veto = None
    if person_veto:
        veto_weights = ROOT / "yolo11n.pt"
        if veto_weights.exists():
            veto = PersonVeto(veto_weights)
            print("person veto: on (COCO yolo11n.pt)")
        else:
            print(f"person veto: OFF, {veto_weights} not found")

    def infer(img: np.ndarray) -> dict:
        if args.model == "yolo":
            return predict_yolo(model, [img], conf=min(conf, 0.05))[0]
        return predict_frcnn(model, [img])[0]

    # warm-up: first inference pays for cudnn autotuning and kernel load
    for _ in range(3):
        infer(frame)

    writer = None
    if args.no_display:
        out_path = Path(args.out or ROOT / "demo" / "demo_annotated.mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        print(f"writing annotated video to {out_path}")

    rec_writer = None
    rec_dir = Path(args.record_dir)
    rec_dir.mkdir(parents=True, exist_ok=True)

    lat_win: deque[float] = deque(maxlen=100)
    loop_win: deque[float] = deque(maxlen=100)
    n = 0
    n_drawn = 0
    n_frames_with = 0
    t_start = time.perf_counter()
    t_prev = t_start

    while True:
        if n:
            ok, frame = cap.read()
            if not ok:
                print("end of stream")
                break
        n += 1
        now = time.perf_counter()
        loop_win.append((now - t_prev) * 1000.0)
        t_prev = now

        t0 = time.perf_counter()
        raw = infer(frame)
        dt = (time.perf_counter() - t0) * 1000.0
        lat_win.append(dt)

        pred = filter_pred(raw, conf, classes, max_aspect, min_area, frame_area, roi_y0)
        n_vetoed = 0
        if veto is not None:
            pred, n_vetoed = veto.apply(pred, veto.persons(frame), frame_area)
        pred = stabiliser(pred)
        n_drawn += len(pred["scores"])
        if len(pred["scores"]):
            n_frames_with += 1

        if rec_writer is not None:
            rec_writer.write(frame)  # raw, no boxes burned in

        annotated = draw(frame.copy(), pred, roi_y0)
        ordered = sorted(lat_win)
        mean_ms = sum(lat_win) / len(lat_win)
        stats = {
            "loop_fps": 1000.0 / max(sum(loop_win) / len(loop_win), 1e-6),
            "model_fps": 1000.0 / max(mean_ms, 1e-6),
            "ms": mean_ms,
            "p50": ordered[len(ordered) // 2],
            "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
            "n_det": int(len(pred["scores"])),
            "vetoed": n_vetoed,
            "conf": conf,
            "safe": args.demo_safe,
            "recording": rec_writer is not None,
        }
        hud(annotated, stats, "YOLO11n" if args.model == "yolo" else "Faster R-CNN")

        if writer is not None:
            writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Litter detection - q quit / s snapshot / r record", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                p = rec_dir / f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                cv2.imwrite(str(p), annotated)
                print(f"snapshot -> {p}")
            elif key == ord("r"):
                if rec_writer is None:
                    p = rec_dir / f"campus_walk_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    rec_writer = cv2.VideoWriter(
                        str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H)
                    )
                    print(f"recording raw frames -> {p}")
                else:
                    rec_writer.release()
                    rec_writer = None
                    print("recording stopped")

        if args.max_frames and n >= args.max_frames:
            print(f"reached --max-frames {args.max_frames}")
            break

    cap.release()
    if writer is not None:
        writer.release()
    if rec_writer is not None:
        rec_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    el = time.perf_counter() - t_start
    ordered = sorted(lat_win)
    print(
        f"\n{n} frames in {el:.1f}s  loop {n / el:.1f} FPS  "
        f"model {1000.0 / (sum(lat_win) / len(lat_win)):.1f} FPS  "
        f"inference mean {sum(lat_win) / len(lat_win):.1f} ms  "
        f"p95 {ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]:.1f} ms"
    )
    print(
        f"boxes drawn: {n_drawn} total, {n_drawn / max(n, 1):.2f} per frame; "
        f"{n_frames_with} of {n} frames had at least one"
    )


if __name__ == "__main__":
    main()
