# Speed benchmark

Device: cuda · 100 timed iterations, 10 warm-up discarded · batch 1 · FP32 · 640 px

| | YOLO11n | Faster R-CNN |
|---|---|---|
| mean ms / image | 12.34 | 21.17 |
| p50 ms | 9.69 | 19.76 |
| p95 ms | 44.01 | 38.27 |
| FPS | 81.0 | 47.2 |
| parameters | 2606098 | 18980254 |
| weights (MB) | 5.5 | 76.11 |
| speed-up | 1.72x | 1.00x |
