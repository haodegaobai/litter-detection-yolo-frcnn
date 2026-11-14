# Campus footage: model response without labels

55 frames decoded at 2 fps from four phone clips.

| threshold | campus boxes/frame | TACO test boxes/frame | ratio |
|---|---|---|---|
| 0.25 | 0.036 | 2.049 | 56.9x |
| 0.50 | 0.018 | 1.204 | 66.9x |

Detections on campus footage at 0.25: 2 in 55 frames. Per class: other 1, plastic 1

This is not a mAP. It measures how often the model fires, not how right it is.
