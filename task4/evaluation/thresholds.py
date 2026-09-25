"""Validation-calibrated rejection threshold (known CIFAR-10 validation data only).

tau = 95th percentile of u(x) over the CIFAR-10 validation split;
accept x as known iff u(x) <= tau  (targets 95 % known acceptance).
"""
from __future__ import annotations

import numpy as np

TARGET_TPR = 0.95


def calibrate_threshold(u_val: np.ndarray, tpr: float = TARGET_TPR) -> float:
    u_val = np.asarray(u_val, dtype=np.float64)
    if u_val.size == 0:
        raise ValueError("empty validation scores")
    return float(np.quantile(u_val, tpr))


def accept(u: np.ndarray, tau: float) -> np.ndarray:
    return np.asarray(u) <= tau
