# DEVIATIONS — FaceLiVTv2

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train data | Glint360K | CelebA | Large FR corpus not local |
| D2 | Variant | S primary tables | XS default | Faster lab twin |
| D3 | CosFace (s,m) | s=64-scale PartialFC | s=16, m=0.25 | CelebA CosFace stability |
| D4 | Epochs / batch | 50 ep, bs 1024 | 12 ep, bs 128 | Full-train budget |
| D5 | Eval | + AgeDB, CFP-FP, IJB | LFW + CALFW + CPLFW | Available locally |
| D6 | Reparameterize deploy | Optional deploy() | Train-mode blocks | Eval without reparam |
| D7 | LR schedule | Polynomial decay, power not stated; lr 6e-3 at batch 1,026 | Polynomial decay with power 1.0 (`lr_poly_power`); lr 1e-3 at batch 128 | Power not given; lr kept for the smaller batch |
