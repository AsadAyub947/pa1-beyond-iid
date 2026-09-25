"""
DAN-DG: same MMD mechanism as Task 2's DAN, but applied pairwise across the
THREE OBSERVED SOURCE DOMAINS ONLY (Photo, Art Painting, Cartoon) -- Sketch is
never involved. The kernel construction (sum of 3 RBF kernels, bandwidths
0.5x/1x/2x the median pairwise squared distance) is intentionally identical
to Task 2's `compute_mmd2` so that Task 2 vs. Task 3's DAN variants differ
ONLY in what data they see (labeled target vs. no target at all), not in the
discrepancy measure itself -- this is exactly what Research Question 4 asks
you to compare.
"""
import itertools

import torch

from methods.erm import erm_classification_loss

BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    sq_norms = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq_norms + sq_norms.t() - 2.0 * (x @ x.t())
    return dists.clamp_min(0.0)


def _median_heuristic_bandwidth(combined: torch.Tensor) -> torch.Tensor:
    n = combined.shape[0]
    sq_dists = _pairwise_sq_dists(combined)
    mask = ~torch.eye(n, dtype=torch.bool, device=combined.device)
    median = sq_dists[mask].median()
    return median.clamp_min(1e-8)


def compute_mmd2(feat_a: torch.Tensor, feat_b: torch.Tensor,
                  bandwidth_multipliers=BANDWIDTH_MULTIPLIERS) -> torch.Tensor:
    """Identical formula to Task 2's methods/dan.py::compute_mmd2 -- do not
    let these two implementations drift apart, since the comparison in
    Research Question 4 depends on them being the same discrepancy measure."""
    combined = torch.cat([feat_a, feat_b], dim=0)
    median_sq_dist = _median_heuristic_bandwidth(combined)
    bandwidths = [median_sq_dist * m for m in bandwidth_multipliers]

    def multi_kernel(a, b):
        sq_norms_a = (a ** 2).sum(dim=1, keepdim=True)
        sq_norms_b = (b ** 2).sum(dim=1, keepdim=True)
        sq_dists = (sq_norms_a + sq_norms_b.t() - 2.0 * (a @ b.t())).clamp_min(0.0)
        k = 0.0
        for bw in bandwidths:
            k = k + torch.exp(-sq_dists / bw)
        return k

    k_aa = multi_kernel(feat_a, feat_a)
    k_bb = multi_kernel(feat_b, feat_b)
    k_ab = multi_kernel(feat_a, feat_b)
    return k_aa.mean() + k_bb.mean() - 2.0 * k_ab.mean()


class DANDGMethod:
    def __init__(self, backbone, head, device, lambda_dg: float = 1.0):
        self.backbone = backbone
        self.head = head
        self.device = device
        self.lambda_dg = lambda_dg  # controlled study sweeps this over {0.1, 1, 10}

    def extra_parameters(self):
        return []

    def step(self, optimizer, source_batches: dict) -> dict:
        optimizer.zero_grad()
        cls_loss, feats_by_domain, logits, y = erm_classification_loss(
            self.backbone, self.head, source_batches, self.device)

        domains = list(feats_by_domain.keys())
        pair_mmds = []
        for a, b in itertools.combinations(domains, 2):
            pair_mmds.append(compute_mmd2(feats_by_domain[a], feats_by_domain[b]))
        mean_pairwise_mmd = torch.stack(pair_mmds).mean()

        loss = cls_loss + self.lambda_dg * mean_pairwise_mmd
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(), "cls_loss": cls_loss.item(),
            "mean_pairwise_mmd": mean_pairwise_mmd.item(),
            **{f"mmd_{a}_{b}": m.item() for (a, b), m in zip(itertools.combinations(domains, 2), pair_mmds)},
        }
