# Domain shift — negative mode (55 frames)

Litter-free frames. Every detection is a false positive; the metric is false positives per frame, not mAP.

| model | conf 0.25 | conf 0.40 | conf 0.55 | conf 0.70 |
|---|---|---|---|---|
| YOLO11n | 0.036 | 0.036 | 0.0 | 0.0 |
| FasterRCNN | 0.236 | 0.073 | 0.036 | 0.0 |

Per class at conf 0.25:

| model | plastic | paper_cardboard | metal | glass | cigarette | other |
|---|---|---|---|---|---|---|
| YOLO11n | 1 | 0 | 0 | 0 | 0 | 1 |
| FasterRCNN | 12 | 1 | 0 | 0 | 0 | 0 |
