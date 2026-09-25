"""
Cosine stability I_T of a backbone's representation between clean and
transformed images (see assignment Section 6, Representation Analysis).
"""
import torch


def cosine_stability(feat_clean: torch.Tensor, feat_transformed: torch.Tensor) -> float:
    """
    I_T = (1/N) * sum_i [ f(x_i)^T f(T(x_i)) / (||f(x_i)|| * ||f(T(x_i))||) ]

    feat_clean, feat_transformed: (N, D) tensors of features from the SAME
    backbone, for the SAME N images, paired clean vs. transformed.
    """
    assert feat_clean.shape == feat_transformed.shape
    a = feat_clean / feat_clean.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = feat_transformed / feat_transformed.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    cos_sim = (a * b).sum(dim=-1)
    return float(cos_sim.mean().item())


def representation_stability_report(feat_clean: torch.Tensor,
                                     transformed_feats: dict) -> dict:
    """transformed_feats: {condition_name: (N,D) tensor paired with feat_clean}"""
    return {name: cosine_stability(feat_clean, feats)
            for name, feats in transformed_feats.items()}
