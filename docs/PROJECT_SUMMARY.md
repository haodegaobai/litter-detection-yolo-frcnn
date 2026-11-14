# Litter Detection on TACO: YOLO11n vs Faster R-CNN

**Tong Yanhao** · 2026

## Problem

Can a small one-stage detector support a live campus-litter demo while a
two-stage detector provides an offline accuracy reference? The project compares
YOLO11n and Faster R-CNN on TACO under one evaluation protocol, then measures
what happens when the models leave the dataset and see a classroom through a
webcam.

## Data

TACO contains 1500 images, 4784 bounding boxes, and 60 fine-grained litter
categories. The long tail makes 60-class training unlearnable at this scale, so
the categories were folded into six material superclasses: plastic,
paper/cardboard, metal, glass, cigarette, and other. The rule was fixed before
training and retained all 4784 annotations.

The dataset has no official split, so a stratified 70/15/15 image-level split was
created once with seed 42: 1050 train images / 3328 objects, 225 validation /
692, and 225 test / 764.

## Method

Both models read the same prepared COCO tree. Five settings were pinned:

- 640 px input. torchvision's default is 800 px on the short side.
- NMS IoU 0.5.
- One mAP implementation: `torchmetrics.MeanAveragePrecision`, COCO-style.
  YOLO's built-in `val()` was not used.
- Validation-only checkpoint selection.
- Batch 1, FP32 timing with 10 warm-up iterations, 100 timed, and CUDA sync.

YOLO11n was fine-tuned for up to 100 epochs with early stopping. It peaked at
epoch 56 and stopped at epoch 76. Faster R-CNN used MobileNetV3-Large FPN, SGD at
lr 0.0015, batch 1, frozen BatchNorm, and 12 epochs. A first Faster R-CNN
schedule at lr 0.005 peaked on its first epoch and was discarded; its test result
is retained for audit but is not the reported model.

## Results

| Metric | YOLO11n | Faster R-CNN |
|---|---:|---:|
| mAP@0.5 | **0.1806** | 0.1615 |
| mAP@0.5:0.95 | **0.1344** | 0.0974 |
| mAP small (<32 px) | **0.0551** | 0.0003 |
| Latency, 640 px, batch 1 | **12.34 ms** | 21.17 ms |
| Throughput | **81.0 FPS** | 47.2 FPS |
| Parameters | **2.61 M** | 18.98 M |

YOLO11n is both faster and more accurate on this split. The result is not
uniform: the advantage is small at IoU 0.5 and larger on the averaged metric and
large objects. The comparison is also asymmetric in training compute: Faster
R-CNN at batch 1 is much slower per epoch, so YOLO received more epochs. That
caveat favours YOLO and is stated rather than hidden.

## What the errors show

At confidence 0.25 and IoU 0.5, YOLO produces 190 true positives, 162 background
false positives, 109 wrong-class detections, and 574 missed ground-truth boxes.
Faster R-CNN produces 159 true positives, 107 background false positives, 124
wrong-class detections, and 605 misses. YOLO is more willing to detect; the
two-stage model is more conservative.

Small objects dominate the loss. YOLO recalls 8.6% of objects below 32 px;
Faster R-CNN recalls 23.3%, but its small-object AP is near zero because its
detections are poorly ranked.

Some background false positives are real litter that TACO never annotated. The
24 highest-scoring background detections were cropped and inspected: 21 are
visibly real litter. That means precision on this test set is understated, and
hard-negative mining on those detections would be counterproductive until they
are labelled.

## Demo engineering

The live demo uses a constrained `--demo-safe` mode because the model is trained
on outdoor litter photographs and labels blackboard strokes as cigarettes in a
classroom. The mode raises confidence to 0.60, keeps only the three classes that
trained, and applies aspect-ratio, area, and two-frame stability filters.

The cost is measured on the same test predictions, not hand-waved:

| Configuration | YOLO P | YOLO R | R-CNN P | R-CNN R |
|---|---:|---:|---:|---:|
| research, 3 classes, conf 0.25 | 0.5000 | 0.3922 | 0.4099 | 0.3420 |
| demo-safe, 3 classes, conf 0.60 | 0.6429 | 0.2745 | 0.7209 | 0.1351 |

On 55 litter-free campus frames, YOLO produces 0.036 false positives per frame
at confidence 0.25, against 0.236 for Faster R-CNN. This measures the classroom
failure mode, but it is not an out-of-domain mAP: those frames have no litter
boxes to match against.

## Limitations

- The two models did not receive equal training compute.
- Small objects remain the dominant failure mode.
- Glass and cigarette are effectively untrained.
- The image-level split may contain near-duplicate scenes from the same
  photographer.
- Positive out-of-domain accuracy needs a hand-corrected campus test set.
- This is a research comparison, not a production detector.

## Next steps

Hand-label a positive campus set, promote YOLO11s (0.1945 mAP@0.5 at 10.63 ms on
the same test split), add labelled hard negatives, and investigate the long-tail
classes with more data rather than a lower confidence threshold.
