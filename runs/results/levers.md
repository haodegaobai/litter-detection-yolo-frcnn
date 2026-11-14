# Three levers, measured on the test split

Same split, same metric code, same timing protocol. The headline comparison in the results stays YOLO11n @640 against Faster R-CNN @640; this table only answers what more would buy.

| variant | mAP@0.5 | Δ | mAP@0.5:0.95 | mAP small | FP @0.5 | ms/img | speed |
|---|---|---|---|---|---|---|---|
| baseline · yolo11n · 640 | 0.1806 | +0.0 | 0.1344 | 0.0551 | 123 | 10.45 | 1.0x |
| resolution · yolo11n · 960 | 0.168 | -0.0126 | 0.1201 | 0.0332 | 114 | 10.96 | 1.05x |
| negatives · yolo11n · 640 | 0.1807 | +0.0001 | 0.1347 | 0.044 | 111 | 10.01 | 0.96x |
| capacity · yolo11s · 640 | 0.1945 | +0.0139 | 0.1479 | 0.0493 | 91 | 10.63 | 1.02x |

Δ and speed are relative to the baseline row.
