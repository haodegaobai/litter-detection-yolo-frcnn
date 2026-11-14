# Deployment configuration: what the demo costs

Re-scored from the saved test predictions. The geometric demo filters (ROI band, aspect ratio, minimum area, stability) are **not** included -- they depend on camera geometry the test split does not share.

| configuration | YOLO P | YOLO R | YOLO F1 | R-CNN P | R-CNN R | R-CNN F1 |
|---|---|---|---|---|---|---|
| research · 6 classes · conf 0.25 | 0.4121 | 0.2487 | 0.3102 | 0.4066 | 0.2081 | 0.2753 |
| research · 6 classes · conf 0.60 | 0.5845 | 0.1675 | 0.2604 | 0.7209 | 0.0812 | 0.1459 |
| research · 3 classes · conf 0.25 | 0.5 | 0.3922 | 0.4396 | 0.4099 | 0.342 | 0.3729 |
| demo-safe · 3 classes · conf 0.40 | 0.5861 | 0.3486 | 0.4372 | 0.5735 | 0.2549 | 0.3529 |
| demo-safe · 3 classes · conf 0.55 | 0.6239 | 0.2963 | 0.4018 | 0.6724 | 0.1699 | 0.2713 |
| demo-safe · 3 classes · conf 0.60 | 0.6429 | 0.2745 | 0.3847 | 0.7209 | 0.1351 | 0.2275 |
| demo-safe · 3 classes · conf 0.70 | 0.6644 | 0.2157 | 0.3257 | 0.7708 | 0.0806 | 0.146 |

Reading it: moving from the research setting to the demo setting raises precision (fewer junk boxes on screen) and lowers recall (fewer real objects found). Both are reported; neither is hidden.
