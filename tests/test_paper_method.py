"""FaceLiVTv2 architecture, reparameterisation and training recipe (Secs. 3-4)."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from facelivt.eval import best_acc_eer
from facelivt.models import facelivtv2_backbone as backbone
from facelivt.models.facelivt_net import CosFaceHead, build_facelivt, count_params
from facelivt.models.facelivtv2_backbone import reparameterize

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", ["xs", "s", "m"])
def test_deployed_parameter_counts_match_table_3(targets, variant):
    net = reparameterize(build_facelivt(10, variant=variant).backbone.eval())
    params = count_params(net) / 1e6
    assert params == pytest.approx(targets["params_m"][variant], abs=targets["params_tolerance_m"])


def test_xs_architecture_matches_table_3(targets):
    arch = targets["xs_architecture"]
    source = inspect.getsource(backbone.facelivtv2_xs)
    assert str(arch["dims"]).replace(" ", "") in source.replace(" ", "")
    assert str(arch["depths"]).replace(" ", "") in source.replace(" ", "")
    assert f"final_feature_dim={arch['head_channels']}" in source.replace(" ", "")
    for mixer in arch["mixers"]:
        assert f'"{mixer}"' in source


def test_reparameterisation_keeps_the_embedding(targets):
    """Sec. 3: structural reparameterisation merges branches without changing the output."""
    torch.manual_seed(0)
    net = build_facelivt(10, variant="xs").backbone.eval()
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        before = net(x)
        after = reparameterize(net)(x)
    assert before.shape == (2, targets["xs_architecture"]["embed_dim"])
    assert torch.allclose(before, after, atol=1e-3)


def test_cosface_subtracts_the_margin_from_the_target_class_only():
    head = CosFaceHead(2, 3, s=10.0, m=0.25)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3, 2))
    logits = head(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))
    assert logits[0].tolist() == pytest.approx([10.0 * (1 - 0.25), 0.0, 0.0], abs=1e-5)


def test_training_uses_adamw_with_polynomial_decay(default_cfg, targets):
    """Sec. 4.1.1: AdamW with a polynomial-decay scheduler and l2 = 1e-2."""
    source = (ROOT / "facelivt" / "train.py").read_text(encoding="utf-8")
    assert "AdamW" in source
    assert "PolynomialLR" in source or "polynomial" in source.lower()
    assert default_cfg["train"]["weight_decay"] == targets["weight_decay"]


def test_eer_is_where_false_match_and_false_non_match_rates_meet():
    """The EER is the error rate where FMR equals FNMR, not the lowest half total error rate."""
    scores = np.array([0.6, 0.7, 0.8, 0.9, 0.0, 0.1, 0.2, 0.65])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    assert best_acc_eer(scores, labels)[1] == pytest.approx(0.25, abs=0.01)
