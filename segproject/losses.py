"""Segmentation losses (distribution-, region-, and metric-surrogate-based) and the composite loss
L_total = sum_k lambda_k * L_k."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import IGNORE


def _prep(logits, y, nc):
    valid = y != IGNORE
    p = logits.float().softmax(1)
    oh = F.one_hot(torch.where(valid, y, torch.zeros_like(y)), nc).permute(0, 3, 1, 2).float()
    m = valid.unsqueeze(1).float()
    return p * m, oh * m


def ce_loss(lg, y, nc, weight=None):
    return F.cross_entropy(lg.float(), y, weight=weight, ignore_index=IGNORE)


def focal_loss(lg, y, nc, gamma=2.0):
    valid = y != IGNORE
    logp = F.log_softmax(lg.float(), 1)
    lp = logp.gather(1, torch.where(valid, y, torch.zeros_like(y)).unsqueeze(1)).squeeze(1)
    loss = -((1 - lp.exp()) ** gamma) * lp
    return (loss * valid).sum() / valid.sum().clamp(min=1)


def dice_loss(lg, y, nc, eps=1.0):
    p, oh = _prep(lg, y, nc)
    inter = (p * oh).sum((0, 2, 3)); den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    d = (2 * inter + eps) / (den + eps)
    present = oh.sum((0, 2, 3)) > 0
    return 1 - d[present].mean()


def tversky_loss(lg, y, nc, alpha=0.3, beta=0.7, eps=1.0):
    p, oh = _prep(lg, y, nc)
    tp = (p * oh).sum((0, 2, 3)); fp = (p * (1 - oh)).sum((0, 2, 3)); fn = ((1 - p) * oh).sum((0, 2, 3))
    t = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    present = oh.sum((0, 2, 3)) > 0
    return 1 - t[present].mean()


def class_weights(freq):
    """ENet-style weights 1/ln(1.02 + p_c)."""
    p = freq / freq.sum()
    return torch.tensor(1.0 / np.log(1.02 + p), dtype=torch.float32)


def parse_loss(s):
    """'ce:1,dice:0.5' -> {'ce':1.0,'dice':0.5}"""
    out = {}
    for tok in s.split(","):
        k, v = tok.split(":")
        out[k.strip()] = float(v)
    return out


class Composite(nn.Module):
    NAMES = ("ce", "wce", "focal", "dice", "tversky", "lovasz")

    def __init__(self, weights, nc, class_w=None):
        super().__init__()
        for k in weights:
            assert k in self.NAMES, f"unknown loss {k}"
        self.w, self.nc, self._lov = weights, nc, None
        self.register_buffer("cw", class_w if class_w is not None else torch.ones(nc))

    def term(self, k, lg, y):
        if k == "ce": return ce_loss(lg, y, self.nc)
        if k == "wce": return ce_loss(lg, y, self.nc, self.cw)
        if k == "focal": return focal_loss(lg, y, self.nc)
        if k == "dice": return dice_loss(lg, y, self.nc)
        if k == "tversky": return tversky_loss(lg, y, self.nc)
        if k == "lovasz":
            if self._lov is None:
                from segmentation_models_pytorch.losses import LovaszLoss
                self._lov = LovaszLoss("multiclass", per_image=False, ignore_index=IGNORE)
            return self._lov(lg.float(), y)

    def forward(self, lg, y):
        return sum(lam * self.term(k, lg, y) for k, lam in self.w.items() if lam > 0)
