# FIDELITY_AUDIT — FaceLiVTv2

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Lite MHLA + RepMix + GDConv | 95% | Upstream backbone |
| 2 | XS/S/M/L factory | 95% | XS default (D2) |
| 3 | CosFace head | 90% | D3 scale |
| 4 | LFW family eval | 92% | IJB omitted (D5) |
| 5 | Size gate + publish | 95% | Stop |

Confidence vs paper structure: **~93%**.

## Measured run (full)

- CelebA+LFW+CALFW+CPLFW ≈ 4.09 GiB → `full_train`
- XS + CosFace s=16/m=0.25; 12 epochs; loss 10.74 → 3.69; **2570 s**
- Params (backbone+head): 6,641,024
- LFW **68.98%**; CALFW **56.82%**; CPLFW **56.33%** (6000 pairs)
