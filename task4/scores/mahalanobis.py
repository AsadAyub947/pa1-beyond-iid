"""Class-conditional Mahalanobis distance with one shared *diagonal* covariance.

u_Mah(x) = min_c (f(x) - mu_c)^T Sigma^{-1} (f(x) - mu_c)

mu_c and Sigma are estimated from the **unaugmented CIFAR-10 training**
features (the 90 % optimisation split) of the frozen model. Sigma is the
pooled within-class variance of each feature dimension (tied across classes,
off-diagonals set to zero), with 1e-6 added to every diagonal entry.
"""
from __future__ import annotations

import numpy as np


class MahalanobisScorer:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.means: np.ndarray | None = None  # [C, D]
        self.var: np.ndarray | None = None    # [D]

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisScorer":
        f = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels)
        classes = np.unique(y)
        self.means = np.stack([f[y == c].mean(axis=0) for c in classes])
        centred = f - self.means[np.searchsorted(classes, y)]
        self.var = (centred ** 2).mean(axis=0) + self.eps  # shared diagonal covariance
        return self

    def distances(self, features: np.ndarray, chunk: int = 8192) -> np.ndarray:
        """Squared Mahalanobis distance to every class mean, shape [N, C]."""
        assert self.means is not None, "call fit() first"
        inv_sd = 1.0 / np.sqrt(self.var)
        m = self.means * inv_sd
        m2 = (m ** 2).sum(1)
        out = []
        f_all = np.asarray(features, dtype=np.float64)
        for s in range(0, len(f_all), chunk):
            f = f_all[s:s + chunk] * inv_sd
            d = (f ** 2).sum(1, keepdims=True) - 2.0 * f @ m.T + m2[None]
            out.append(np.maximum(d, 0.0))
        return np.concatenate(out) if out else np.zeros((0, len(m)))

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.distances(features).min(axis=1)


def mahalanobis_score(train_features, train_labels, features, eps: float = 1e-6) -> np.ndarray:
    return MahalanobisScorer(eps).fit(train_features, train_labels).score(features)
