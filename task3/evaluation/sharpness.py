"""
Common local-sharpness diagnostic (assignment Step 4), applied identically to
ERM, DAN-DG, and SAM so the comparison is fair: a FIXED validation batch (32
examples per source domain, seed 6304), model in eval mode, one normalized
gradient-ascent perturbation of radius 0.05, measuring the resulting increase
in cross-entropy loss.
"""
import numpy as np
import torch
import torch.nn.functional as F

SEED = 6304
SHARPNESS_RHO = 0.05
N_PER_DOMAIN = 32


def build_fixed_sharpness_batch(val_source_datasets: dict, n_per_domain: int = N_PER_DOMAIN,
                                 seed: int = SEED, device=None):
    """val_source_datasets: {domain_name: PACSDomainDataset} for the three
    source validation sets (NOT loaders -- we need direct, seeded indexing).
    Returns (x, y) tensors on `device`, fixed once and reused for every model
    evaluated with this diagnostic."""
    rng = np.random.RandomState(seed)
    xs, ys = [], []
    for domain in sorted(val_source_datasets.keys()):
        ds = val_source_datasets[domain]
        n = min(n_per_domain, len(ds))
        idx = rng.choice(len(ds), size=n, replace=False)
        for i in idx:
            x, y = ds[i]
            xs.append(x)
            ys.append(y)
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def compute_sharpness(backbone, head, fixed_x: torch.Tensor, fixed_y: torch.Tensor,
                       rho: float = SHARPNESS_RHO) -> float:
    """Delta_sharp = L(theta + epsilon) - L(theta),
       epsilon = rho * grad_theta(L(theta)) / ||grad_theta(L(theta))||_2

    The model is placed in eval() mode (BatchNorm uses running stats, no
    dropout) but gradients w.r.t. parameters are still computed normally --
    eval() only affects module forward behavior, not autograd."""
    backbone.eval()
    head.eval()

    params = [p for p in list(backbone.parameters()) + list(head.parameters()) if p.requires_grad]
    for p in params:
        p.grad = None

    with torch.enable_grad():
        logits_clean = head(backbone(fixed_x))
        loss_clean = F.cross_entropy(logits_clean, fixed_y)
        loss_clean.backward()

    with torch.no_grad():
        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]), 2)
        e_ws = []
        for p in params:
            if p.grad is None:
                e_ws.append(None)
                continue
            e_w = rho * p.grad / (grad_norm + 1e-12)
            p.add_(e_w)
            e_ws.append(e_w)

        logits_perturbed = head(backbone(fixed_x))
        loss_perturbed = F.cross_entropy(logits_perturbed, fixed_y)

        # restore original parameters
        for p, e_w in zip(params, e_ws):
            if e_w is not None:
                p.sub_(e_w)

    for p in params:
        p.grad = None

    delta_sharp = float(loss_perturbed.item() - loss_clean.item())
    return delta_sharp
