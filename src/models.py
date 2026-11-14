"""Model loading and prediction for both detectors, behind one interface.

Everything that touches "how a model turns an image into boxes" lives here, so
that run_eval.py, benchmark.py, eval_campus.py and live_demo.py cannot drift
apart. If the NMS threshold changes, it changes once.

Fairness settings pinned in this file
-------------------------------------
* input 640 px for both (Faster R-CNN's internal resize is forced to 640/640;
  its default is 800 on the short side, which would silently give it 1.6x more
  pixels than YOLO and make the speed comparison meaningless)
* NMS IoU 0.5 for both
* at most 100 detections per image for both
* a low score floor for both, so torchmetrics sees the full precision/recall
  curve instead of a pre-truncated one

Coordinate convention: both models return boxes in the *original* pixel frame of
the image handed in -- YOLO rescales its letterboxed output, torchvision's
GeneralizedRCNNTransform resizes boxes back in postprocess. Ground truth from
data_utils is in the same frame, so IoU matching is valid without extra work.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from data_utils import NUM_CLASSES

IMGSZ = int(os.environ.get("IMGSZ", 640))
CONF_FLOOR = 0.001
NMS_IOU = 0.5
MAX_DET = 100
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and os.environ.get("DEVICE", "0") != "cpu" else "cpu"
)


# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
def load_yolo(weights: Path):
    from ultralytics import YOLO

    return YOLO(str(weights))


@torch.no_grad()
def predict_yolo(model, sources, conf: float = CONF_FLOOR) -> list[dict]:
    """sources: list of paths or BGR numpy frames. Returns one dict per source."""
    results = model.predict(
        source=sources,
        imgsz=IMGSZ,
        conf=conf,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=0 if DEVICE.type == "cuda" else "cpu",
        verbose=False,
    )
    out = []
    for r in results:
        b = r.boxes
        if b is None or len(b) == 0:
            out.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )
            continue
        out.append(
            {
                "boxes": b.xyxy.detach().cpu().float(),
                "scores": b.conf.detach().cpu().float(),
                "labels": b.cls.detach().cpu().long(),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Faster R-CNN
# --------------------------------------------------------------------------- #
def load_frcnn(weights: Path):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)

    model.transform.min_size = (IMGSZ,)
    model.transform.max_size = IMGSZ
    model.roi_heads.score_thresh = CONF_FLOOR
    model.roi_heads.nms_thresh = NMS_IOU
    model.roi_heads.detections_per_img = MAX_DET
    model.to(DEVICE).eval()
    return model


def _to_tensor_rgb(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC -> RGB float CHW in 0..1 (torchvision normalises internally)."""
    rgb = image_bgr[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)


FRCNN_EVAL_CHUNK = int(os.environ.get("FRCNN_EVAL_CHUNK", 4))


@torch.no_grad()
def predict_frcnn(model, images_bgr: list[np.ndarray], chunk: int = FRCNN_EVAL_CHUNK) -> list[dict]:
    """Predict in chunks.

    Chunking does not change any per-image result (the model is in eval mode and
    BatchNorm is frozen, so images never interact), it only bounds peak VRAM.
    Feeding a whole split at once makes the ROI head allocate tens of GiB and
    OOMs on a 6 GB laptop GPU -- which is exactly what happened the first time
    this ran over the 119-image test split.
    """
    if not images_bgr:
        return []
    outputs = []
    for i in range(0, len(images_bgr), max(1, chunk)):
        batch = [_to_tensor_rgb(im).to(DEVICE) for im in images_bgr[i : i + chunk]]
        outputs.extend(model(batch))
        del batch
    out = []
    for o in outputs:
        keep = o["labels"] > 0  # 0 is background in torchvision detectors
        out.append(
            {
                "boxes": o["boxes"][keep].detach().cpu().float(),
                "scores": o["scores"][keep].detach().cpu().float(),
                "labels": (o["labels"][keep] - 1).detach().cpu().long(),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def read_bgr(path: Path) -> np.ndarray:
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"cannot read {path}")
    return im


def to_eval_dicts(preds: list[dict]) -> list[dict]:
    return [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in preds]
