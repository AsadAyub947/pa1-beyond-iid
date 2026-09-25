"""OPTIONAL – Reciprocal Point Learning (Chen et al., ECCV 2020).

Adapted from the authors' public implementation (github.com/iCGY96/ARPL,
``loss/RPLoss.py`` + ``loss/Dist.py``). What each part does:

* Reciprocal points – one learnable vector P^k in R^512 per known class
  (``ReciprocalPointHead.points``, init 0.1*N(0,1)), trained jointly with the
  backbone by SGD. P^k represents the *extra-class* space of class k
  ("what class k is not").
* Known-class score – the mean squared Euclidean distance
      d(f(x), P^k) = ||f(x) - P^k||^2 / D
  is used directly as the class-k logit. A feature of class k is pushed *away*
  from P^k: CE(d / T, y) with temperature T = 1 maximises d(f(x), P^y)
  relative to the other distances.
* Open-space regularisation – a learnable radius R (``head.radius``) and
      L_o = mean_i ( d(f(x_i), P^{y_i}) - R )^2            (open_space: mse)
  keep every known embedding at a bounded distance from its reciprocal point,
  so the known feature space cannot expand without limit and the space close
  to the reciprocal points (the open space) stays reserved for unknowns.
  ``open_space: hinge`` uses max(d - R, 0) instead (paper's bounded form).
  Total loss: L = CE(d/T, y) + lambda * L_o, lambda = 0.1.
* Rejection score – u_RPL(x) = -max_k d(f(x), P^k): an input that is not far
  from any reciprocal point is unknown (see ``scores/rpl_distance.py``).
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.vanilla import VanillaMethod
from models.resnet_cifar import FEATURE_DIM, build_resnet18_cifar


class ReciprocalPointHead(nn.Module):
    def __init__(self, feat_dim: int = FEATURE_DIM, num_classes: int = 10, init_scale: float = 0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.points = nn.Parameter(init_scale * torch.randn(num_classes, feat_dim))
        self.radius = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """Distances to every reciprocal point, [N, K] (used as the logits)."""
        ctx = (torch.autocast(device_type=f.device.type, enabled=False)
               if f.device.type in ("cuda", "cpu") else contextlib.nullcontext())
        with ctx:  # distances in float32 even under mixed precision
            f = f.float()
            p = self.points.float()
            d = (f.pow(2).sum(1, keepdim=True) - 2.0 * f @ p.t() + p.pow(2).sum(1)[None])
            return d / float(self.feat_dim)


class RPLMethod(VanillaMethod):
    name = "rpl"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        r = cfg.get("rpl", {})
        self.temp = float(r.get("temperature", 1.0))
        self.lam = float(r.get("lambda_open", 0.1))
        self.open_space = str(r.get("open_space", "mse"))
        self.init_scale = float(r.get("point_init_scale", 0.1))

    def build_model_skeleton(self) -> torch.nn.Module:
        model = build_resnet18_cifar(num_classes=self.num_classes)
        model.fc = ReciprocalPointHead(FEATURE_DIM, self.num_classes, self.init_scale)
        return model

    def build_model(self) -> torch.nn.Module:
        return self.build_model_skeleton()

    def training_loss(self, model, x, y):
        out = model(x)
        dist = out["logits"].float()                    # [B, K] distances
        loss_ce = F.cross_entropy(dist / self.temp, y)
        d_own = dist.gather(1, y.view(-1, 1)).squeeze(1)  # d(f(x), P^y)
        radius = model.fc.radius.float()
        if self.open_space == "hinge":
            loss_open = F.relu(d_own - radius).mean()
        else:
            loss_open = F.mse_loss(d_own, radius.expand_as(d_own))
        loss = loss_ce + self.lam * loss_open
        return loss, {"correct": (dist.argmax(1) == y).sum().item(), "n": y.numel(),
                      "loss_ce": loss_ce.item(), "loss_open": loss_open.item(),
                      "radius": radius.item()}
