"""
Gradient-reversal layer (Ganin et al., 2016) + the shared domain-discriminator
architecture used identically by DANN and CDAN (they differ only in what
feature is fed to it: f for DANN, vec(f (x) p) for CDAN).
"""
import math

import torch
import torch.nn as nn


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverseFn.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    """Stores no state itself; alpha is passed in at call time since it
    changes every training step according to the schedule below."""

    def forward(self, x, alpha: float):
        return grad_reverse(x, alpha)


def grl_alpha_schedule(p: float, max_alpha: float = 1.0) -> float:
    """alpha(p) = 2 / (1 + exp(-10p)) - 1, p in [0, 1] = training progress.
    max_alpha rescales the [0, 1] schedule output -- this is the knob swept
    in the controlled design study (Step 6, DANN variant): reversal strength
    in {0.25, 0.5, 1}."""
    base = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
    return max_alpha * base


class DomainDiscriminator(nn.Module):
    """256-unit hidden layer, ReLU, dropout 0.5, 2-class output -- fixed
    architecture shared by DANN (input_dim=512) and CDAN (input_dim=512*7)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


def cdan_feature(f: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """g(x) = vec(f (x) p), the outer product of the 512-dim feature and the
    7-dim class-probability vector, flattened to a 512*7=3584-dim vector.
    Neither f nor p is detached, per the assignment's requirement (no entropy
    conditioning, no detaching f or p) -- gradients from the domain loss flow
    back into BOTH the backbone (via f) and the classifier head (via p)."""
    b = f.shape[0]
    outer = f.unsqueeze(2) * p.unsqueeze(1)  # (B, feat_dim, num_classes)
    return outer.view(b, -1)
