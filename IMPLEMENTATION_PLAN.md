# Implementation Plan — FaceLiVTv2

1. Refactor upstream `facelivtv2.py` (Lite MHLA + RepMix + GDConv) into package.
2. CosFace head; train XS on CelebA (Glint360K substitute).
3. Eval LFW / CALFW / CPLFW; smoke → size gate → full → publish.
