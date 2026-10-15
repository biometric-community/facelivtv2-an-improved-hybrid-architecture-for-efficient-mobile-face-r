# FaceLiVTv2: Efficient Mobile Face Recognition

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of **FaceLiVTv2** (Lite MHLA + RepMix + GDConv + CosFace) for mobile face recognition.

**Repository:** https://github.com/biometric-community/facelivtv2-an-improved-hybrid-architecture-for-efficient-mobile-face-r

**Paper:** [10.1109/TBIOM.2026.3690540](https://doi.org/10.1109/TBIOM.2026.3690540) · [arXiv:2604.09127](https://arxiv.org/abs/2604.09127)

**Upstream:** [novendrastywn/FaceLiVT](https://github.com/novendrastywn/FaceLiVT)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
