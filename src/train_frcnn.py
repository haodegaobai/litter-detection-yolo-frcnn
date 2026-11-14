"""Fine-tune Faster R-CNN (MobileNetV3-Large FPN) with a custom training loop.

Three details decide whether the comparison with YOLO is fair
------------------------------------------------------------
1. torchvision's detector applies ImageNet normalisation inside
   `GeneralizedRCNNTransform`, so the dataset hands it raw 0-1 floats.
2. That same transform defaults to resizing the short side to 800 px. Left alone,
   Faster R-CNN would see 800 px inputs while YOLO sees 640, and both the mAP and
   the speed comparison would be measuring input resolution instead of
   architecture. It is pinned to 640/640 here and again in models.py.
3. torchvision detectors reserve label 0 for background. Dataset labels (0..5)
   are shifted to 1..6 for training and shifted back after inference. Skipping
   this trains `plastic` -- 44% of all instances -- as background.

BatchNorm is frozen: at batch size 1 (forced by 6 GB of VRAM) the batch
statistics are pure noise, and letting them drift destroys the pre-trained
backbone's calibration within a few epochs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

from common_eval import evaluate_map
from data_utils import NUM_CLASSES, ROOT, load_gt
from models import DEVICE, IMGSZ

EPOCHS = int(os.environ.get("FRCNN_EPOCHS", 25 if DEVICE.type == "cuda" else 4))
PATIENCE = int(os.environ.get("FRCNN_PATIENCE", 8))
LR = float(os.environ.get("FRCNN_LR", 0.005))
# Allows a second schedule to be trained without clobbering the first run's
# weights and history, so both can be compared on the validation split.
OUT_NAME = os.environ.get("FRCNN_OUT", "frcnn")
SEED = 42


class LitterDataset(Dataset):
    def __init__(self, records: list[dict], train: bool = False):
        # torchvision detectors cannot handle images with zero ground truth
        self.records = [r for r in records if r["boxes"].shape[0] > 0]
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        boxes = r["boxes"].clone()
        labels = r["labels"].clone() + 1  # 0 is background in torchvision

        if self.train and torch.rand(1).item() < 0.5:
            w = img.width
            img = TF.hflip(img)
            if boxes.shape[0]:
                boxes = torch.stack(
                    [w - boxes[:, 2], boxes[:, 1], w - boxes[:, 0], boxes[:, 3]], dim=1
                )

        tensor = TF.pil_to_tensor(img).float() / 255.0
        return tensor, {"boxes": boxes, "labels": labels}


def collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def build_model() -> torch.nn.Module:
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=0.05)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    model.transform.min_size = (IMGSZ,)
    model.transform.max_size = IMGSZ
    return model


@torch.no_grad()
def validate(model, loader) -> list[dict]:
    model.eval()
    preds = []
    for images, _ in loader:
        outputs = model([im.to(DEVICE) for im in images])
        for out in outputs:
            keep = out["labels"] > 0
            preds.append(
                {
                    "boxes": out["boxes"][keep].detach().cpu(),
                    "scores": out["scores"][keep].detach().cpu(),
                    "labels": out["labels"][keep].detach().cpu() - 1,
                }
            )
    return preds


def main() -> None:
    torch.manual_seed(SEED)
    out_dir = ROOT / "runs" / OUT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = load_gt("train")
    val_recs = load_gt("valid")
    ds_train = LitterDataset(train_recs, train=True)
    ds_val = LitterDataset(val_recs, train=False)
    val_gt = [
        {"boxes": r["boxes"], "labels": r["labels"]}
        for r in val_recs
        if r["boxes"].shape[0] > 0
    ]

    print(
        f"device={DEVICE} train_imgs={len(ds_train)} val_imgs={len(ds_val)} "
        f"epochs={EPOCHS} imgsz={IMGSZ}",
        flush=True,
    )

    dl_train = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, collate_fn=collate, num_workers=0)

    model = build_model().to(DEVICE)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
            mod.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)

    def lr_lambda(ep: int) -> float:
        if ep == 0:
            return 0.2  # warm-up: the new box head starts from random weights
        if ep < int(EPOCHS * 0.5):
            return 1.0
        if ep < int(EPOCHS * 0.8):
            return 0.5
        return 0.25

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = []
    best_map = -1.0
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for mod in model.modules():  # model.train() re-enables BN momentum
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.eval()
        t0 = time.time()
        running: dict[str, float] = {}
        n_batches = 0

        for images, targets in dl_train:
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            optimizer.step()
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v)
            n_batches += 1
        scheduler.step()

        avg = {k: round(v / max(n_batches, 1), 4) for k, v in running.items()}
        total = round(sum(avg.values()), 4)

        preds = validate(model, dl_val)
        m = evaluate_map(preds, val_gt)
        rec = {
            "epoch": epoch,
            "train_loss": total,
            "loss_parts": avg,
            "val_mAP_0.5": m["mAP_0.5"],
            "val_mAP_0.5:0.95": m["mAP_0.5:0.95"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)

        if m["mAP_0.5"] > best_map:
            best_map = m["mAP_0.5"]
            bad = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  -> new best val mAP@0.5 = {best_map:.4f}", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break

        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best val mAP@0.5 = {best_map:.4f}")


if __name__ == "__main__":
    main()
