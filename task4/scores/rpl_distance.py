"""RPL rejection score (optional extension; Chen et al., 2020).

For RPL the model's "logits" are the distances d(f(x), P^k) of the embedding
to each class's reciprocal point: a *large* distance from P^k is evidence for
class k. An input that is not sufficiently far from *any* reciprocal point is
rejected, so

    u_RPL(x) = -max_k d(f(x), P^k)        (larger = more novel)
"""
import numpy as np


def rpl_score(distances: np.ndarray) -> np.ndarray:
    return -np.asarray(distances, dtype=np.float64).max(axis=1)
