"""
Multi-kernel MMD^2 shared by Task 2 (DAN, source vs target) and Task 3 (DAN-DG,
pairs of source domains) so both tasks use exactly the same discrepancy.

Kernel: sum of three RBF kernels k(x, y) = sum_m exp(-||x - y||^2 / (m * med)),
m in {0.5, 1, 2}, med = median pairwise squared distance in the current
combined batch (off-diagonal). The bandwidth is treated as a constant
(computed without gradient), as in DAN.

Estimator: UNBIASED (U-statistic) by default -- the within-set averages exclude
the diagonal k(x, x) = 3. The biased (V-statistic) estimator used previously
has an expected value of about 2 * (3 - E k) / n even when both sets come from
the same distribution: ~0.16 for 24 vs 24 samples (Task 2) and ~0.47 for
8 vs 8 (each Task 3 pair). That floor dominated the logged MMD, so the curves
could not show alignment, and its gradient rewards pulling same-domain
features together regardless of class. The unbiased estimate is ~0 in
expectation for identical distributions (and can be slightly negative).
"""
import torch

BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def _sq_dists(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a ** 2).sum(1, keepdim=True) + (b ** 2).sum(1, keepdim=True).t() - 2.0 * (a @ b.t())
    return d.clamp_min(0.0)


def median_bandwidth(combined: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        n = combined.shape[0]
        d = _sq_dists(combined, combined)
        off = d[~torch.eye(n, dtype=torch.bool, device=combined.device)]
        return off.median().clamp_min(1e-8)


def _multi_kernel(a, b, bandwidths):
    d = _sq_dists(a, b)
    return sum(torch.exp(-d / bw) for bw in bandwidths)


def _offdiag_mean(k: torch.Tensor) -> torch.Tensor:
    n = k.shape[0]
    return (k.sum() - k.diagonal().sum()) / (n * (n - 1))


def compute_mmd2(feat_a: torch.Tensor, feat_b: torch.Tensor, estimator: str = "unbiased",
                 bandwidth_multipliers=BANDWIDTH_MULTIPLIERS) -> torch.Tensor:
    med = median_bandwidth(torch.cat([feat_a, feat_b], dim=0))
    bws = [med * m for m in bandwidth_multipliers]
    k_aa = _multi_kernel(feat_a, feat_a, bws)
    k_bb = _multi_kernel(feat_b, feat_b, bws)
    k_ab = _multi_kernel(feat_a, feat_b, bws)
    if estimator == "biased":
        return k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean()
    if estimator != "unbiased":
        raise ValueError(f"unknown MMD estimator {estimator!r}")
    return _offdiag_mean(k_aa) + _offdiag_mean(k_bb) - 2.0 * k_ab.mean()
