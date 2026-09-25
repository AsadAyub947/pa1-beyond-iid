"""PROSER placeholder-based detection score (Zhou et al., 2021; reference code).

At test time the reference implementation appends the *strongest* dummy
response to the K known-class logits and applies a softmax over these K+1
values; the probability of the dummy ("unknown") slot is the novelty score:

    u_PROSER(x) = softmax([z_1, ..., z_K, max_c d_c(x)])_{K+1}

The paper calibrates the rejection rule on known validation data by adding a
bias to the dummy output so that 95 % of validation examples stay known. That
rule is equivalent to thresholding the margin ``max_c d_c - max_k z_k`` at its
95th validation percentile, which is also provided (``proser_margin_score``).
"""
import numpy as np
from scipy.special import softmax


def proser_dummy_score(logits: np.ndarray, dummy_logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    d = np.asarray(dummy_logits, dtype=np.float64).max(axis=1, keepdims=True)
    return softmax(np.concatenate([z, d], axis=1), axis=1)[:, -1]


def proser_margin_score(logits: np.ndarray, dummy_logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    d = np.asarray(dummy_logits, dtype=np.float64)
    return d.max(axis=1) - z.max(axis=1)
