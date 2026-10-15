# SOURCE_CODE

Upstream: [novendrastywn/FaceLiVT](https://github.com/novendrastywn/FaceLiVT)

Refactored:

- `backbones/facelivtv2.py` → `facelivt/models/facelivtv2_backbone.py`
- CosFace head added in `facelivt/models/facelivt_net.py`
- Training/eval rewritten for CelebA + LFW/CALFW/CPLFW (InsightFace/DALI stack not vendored)
