"""FaceLiVTv2 backbone + CosFace head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .facelivtv2_backbone import facelivtv2_l, facelivtv2_m, facelivtv2_s, facelivtv2_xs


_VARIANTS = {
    "xs": facelivtv2_xs,
    "s": facelivtv2_s,
    "m": facelivtv2_m,
    "l": facelivtv2_l,
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CosFaceHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int, s: float = 16.0, m: float = 0.25):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        return (logits - one_hot * self.m) * self.s


class FaceLiVTNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        variant: str = "xs",
        s: float = 16.0,
        m: float = 0.25,
    ):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError(variant)
        # upstream uses num_classes as embedding dim (512)
        self.backbone = _VARIANTS[variant](num_classes=512, distillation=False)
        self.head = CosFaceHead(512, n_classes, s=s, m=m)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        emb = self.backbone(x)
        if labels is None:
            return F.normalize(emb, p=2, dim=1)
        return self.head(emb, labels), F.normalize(emb, p=2, dim=1)


def build_facelivt(n_classes: int, variant: str = "xs", **kwargs) -> FaceLiVTNet:
    return FaceLiVTNet(n_classes, variant=variant, **kwargs)
