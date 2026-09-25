"""Maximum Softmax Probability (Hendrycks & Gimpel, 2017).

u_MSP(x) = 1 - max_k softmax(z(x))_k        (larger = more novel)
"""
import numpy as np
from scipy.special import softmax


def msp_score(logits: np.ndarray) -> np.ndarray:
    p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return 1.0 - p.max(axis=1)
