# DEVIATIONS — FaceLiVTv2

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Train data | Glint360K | CelebA | Large FR corpus not local |
| D2 | Variant | S primary tables | XS default | Faster lab twin |
| D3 | CosFace (s,m) | s=64-scale PartialFC | s=16, m=0.25 | CelebA CosFace stability |
| D4 | Epochs / batch | 50 ep, bs 1024 | 12 ep, bs 128 | Full-train budget |
| D5 | Eval | + AgeDB, CFP-FP, IJB | LFW + CALFW + CPLFW | Available locally |
| D6 | Reparameterize deploy | Optional deploy() | Train-mode blocks | Eval without reparam |
