"""Manifold mixup between *different* CIFAR-10 classes (PROSER data placeholders).

    h_i = phi_pre(x_i),  h_j = phi_pre(x_j),  y_i != y_j,  lambda ~ Beta(alpha, alpha)
    h~  = lambda * h_i + (1 - lambda) * h_j

For the CIFAR ResNet-18, ``phi_pre`` is the network up to and including
``layer2``; ``h~`` is passed through ``layer3 -> layer4 -> pool -> heads``.
Only CIFAR-10 training images from the current mini-batch are mixed.
"""
from __future__ import annotations

import torch


def sample_different_class_partners(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For every i pick a random j (uniform) with ``y[j] != y[i]``.

    Returns ``(partner_index, valid_mask)``; ``valid_mask[i]`` is False only if
    no example of another class exists in the batch.
    """
    diff = (y[:, None] != y[None, :]).float()
    valid = diff.sum(1) > 0
    weights = diff.clone()
    weights[~valid] = 1.0  # dummy row so multinomial is defined; masked out below
    partner = torch.multinomial(weights, 1).squeeze(1)
    return partner, valid


def sample_lambda(n: int, alpha: float, per_sample: bool, device) -> torch.Tensor:
    dist = torch.distributions.Beta(torch.tensor(float(alpha)), torch.tensor(float(alpha)))
    lam = dist.sample((n,)) if per_sample else dist.sample().expand(n)
    return lam.to(device)


def manifold_mixup(h: torch.Tensor, y: torch.Tensor, alpha: float = 2.0,
                   per_sample: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix intermediate representations ``h`` [B,C,H,W] across different classes.

    Returns ``(h_mixed[valid], valid_mask)``.
    """
    partner, valid = sample_different_class_partners(y)
    lam = sample_lambda(h.size(0), alpha, per_sample, h.device).to(h.dtype).view(-1, 1, 1, 1)
    h_mix = lam * h + (1.0 - lam) * h[partner]
    return h_mix[valid], valid
