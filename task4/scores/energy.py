"""Energy score (Liu et al., 2020) with temperature T = 1.

u_Energy(x) = -log sum_k exp(z_k(x))        (larger = more novel)
"""
import numpy as np
from scipy.special import logsumexp


def energy_score(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)
