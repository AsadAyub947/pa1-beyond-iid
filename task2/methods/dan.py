"""
DAN-style alignment: penalize Maximum Mean Discrepancy between pooled source
and target 512-dim features, using a sum of three RBF kernels whose
bandwidths are 0.5x, 1x, and 2x the median pairwise squared feature distance
in the CURRENT combined batch (recomputed every step, per the assignment).
"""
import torch

from methods.base import AdaptationMethod

BANDWIDTH_MULTIPLIERS = (0.5, 1.0, 2.0)


def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    # ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi.xj
    sq_norms = (x ** 2).sum(dim=1, keepdim=True)
    dists = sq_norms + sq_norms.t() - 2.0 * (x @ x.t())
    return dists.clamp_min(0.0)


def _median_heuristic_bandwidth(combined: torch.Tensor) -> torch.Tensor:
    n = combined.shape[0]
    sq_dists = _pairwise_sq_dists(combined)
    # exclude the zero diagonal (self-distances) when taking the median
    mask = ~torch.eye(n, dtype=torch.bool, device=combined.device)
    off_diag = sq_dists[mask]
    median = off_diag.median()
    return median.clamp_min(1e-8)  # guard against a degenerate all-identical batch


def compute_mmd2(feat_s: torch.Tensor, feat_t: torch.Tensor,
                  bandwidth_multipliers=BANDWIDTH_MULTIPLIERS) -> torch.Tensor:
    """Biased MMD^2 estimator with a sum-of-RBF-kernels kernel, matching
    ||E_s[phi(F(xs))] - E_t[phi(F(xt))]||^2_H via the kernel trick (the
    kernel sum is never explicitly turned into a feature map phi)."""
    combined = torch.cat([feat_s, feat_t], dim=0)
    median_sq_dist = _median_heuristic_bandwidth(combined)
    bandwidths = [median_sq_dist * m for m in bandwidth_multipliers]

    def multi_kernel(a, b):
        sq_norms_a = (a ** 2).sum(dim=1, keepdim=True)
        sq_norms_b = (b ** 2).sum(dim=1, keepdim=True)
        sq_dists = sq_norms_a + sq_norms_b.t() - 2.0 * (a @ b.t())
        sq_dists = sq_dists.clamp_min(0.0)
        k = 0.0
        for bw in bandwidths:
            k = k + torch.exp(-sq_dists / bw)
        return k

    k_ss = multi_kernel(feat_s, feat_s)
    k_tt = multi_kernel(feat_t, feat_t)
    k_st = multi_kernel(feat_s, feat_t)

    mmd2 = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
    return mmd2


class DANMethod(AdaptationMethod):
    def __init__(self, backbone, head, device, lambda_mmd: float = 1.0):
        super().__init__(backbone, head, device)
        self.lambda_mmd = lambda_mmd

    def compute_step(self, source_batches: dict, target_batch, progress: float) -> dict:
        cls_loss, feat_s, logits_s, y_s = self._source_classification_loss(source_batches)

        x_t, _y_t_UNUSED = target_batch  # target labels must never be used here
        x_t = x_t.to(self.device)
        feat_t = self.backbone(x_t)

        mmd2 = compute_mmd2(feat_s, feat_t)
        loss = cls_loss + self.lambda_mmd * mmd2
        return {"loss": loss, "cls_loss": cls_loss.item(), "mmd": mmd2.item()}
