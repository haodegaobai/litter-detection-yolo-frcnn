# Attribution and third-party notices

This repository is licensed under AGPL-3.0. It uses or redistributes artifacts
from the projects below.

## TACO

- **Dataset:** TACO: Trash Annotations in Context
- **Authors:** Pedro F. Proença and Pedro Simões
- **Annotations:** CC BY 4.0
- **Images:** each image retains the license recorded in the TACO annotation
  file; images without a recorded license default to CC BY 4.0.
- **Website:** http://www.tacodataset.org/
- **Paper:** Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in
  Context for Litter Detection. arXiv:2003.06975.

TACO images are not redistributed in this repository. Figures and demo material
that contain TACO images are derived from the dataset and retain the applicable
TACO image licenses. The dataset download scripts write images under `data/`,
which is excluded from Git.

## Ultralytics YOLO11

- **Project:** Ultralytics YOLO11
- **Authors:** Glenn Jocher and Jing Qiu
- **License:** AGPL-3.0
- **Repository:** https://github.com/ultralytics/ultralytics

The committed YOLO checkpoints are fine-tuned from YOLO11n. Ultralytics states
that YOLO code, architectures, training pipelines, and trained or fine-tuned
models are covered by AGPL-3.0 unless an Enterprise License is obtained. This
repository is released under AGPL-3.0 accordingly.

## torchvision / Faster R-CNN

- **Project:** pytorch/vision
- **License:** BSD 3-Clause
- **Repository:** https://github.com/pytorch/vision

The Faster R-CNN implementation and the MobileNetV3-Large FPN backbone come from
torchvision. The reported Faster R-CNN checkpoint is distributed as a GitHub
Release asset.

## Demo footage and screenshots

The webcam clips and screenshots in `demo/` and `screenshots/` were captured by
the author for this project, unless a frame contains TACO imagery. TACO-derived
frames retain the TACO image licenses above.

## Citation

```bibtex
@article{proenca2020taco,
  title   = {TACO: Trash Annotations in Context for Litter Detection},
  author  = {Proença, Pedro F. and Simões, Pedro},
  journal = {arXiv preprint arXiv:2003.06975},
  year    = {2020}
}

@software{yolo11_ultralytics,
  author  = {Jocher, Glenn and Qiu, Jing},
  title   = {Ultralytics YOLO11},
  year    = {2024},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}

@inproceedings{ren2015faster,
  title     = {Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks},
  author    = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2015}
}
```
