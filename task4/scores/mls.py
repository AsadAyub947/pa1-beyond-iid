"""Maximum Logit Score (Vaze et al., 2022).

u_MLS(x) = -max_k z_k(x)        (larger = more novel)
"""
import numpy as np


def mls_score(logits: np.ndarray) -> np.ndarray:
    return -np.asarray(logits, dtype=np.float64).max(axis=1)
