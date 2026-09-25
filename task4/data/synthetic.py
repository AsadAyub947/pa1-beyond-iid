"""Tiny synthetic stand-ins for CIFAR-10 / CIFAR-100 (smoke tests only).

Enabled with ``--override data.synthetic="{n_train: 2000, n_test: 500}"``.
Never used for real experiments: it only lets the whole pipeline run offline
in a few minutes on a CPU to check that every script works.
"""
from __future__ import annotations

import numpy as np

from data.cifar100_unknowns import FAR_UNKNOWN, NEAR_UNKNOWN

_C10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _prototypes(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 255, size=(n, 4, 4, 3))
    return np.kron(base, np.ones((1, 8, 8, 1)))  # blocky 32x32 patterns


def _render(protos: np.ndarray, labels: np.ndarray, rng, noise: float = 40.0) -> np.ndarray:
    x = protos[labels] + rng.normal(0, noise, size=(len(labels), 32, 32, 3))
    return np.clip(x, 0, 255).astype(np.uint8)


def synthetic_cifar10(train: bool, n_train: int = 2000, n_test: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed + (0 if train else 1))
    n = n_train if train else n_test
    y = np.repeat(np.arange(10), int(np.ceil(n / 10)))[:n]
    rng.shuffle(y)
    x = _render(_prototypes(10, seed), y, rng)
    return x, y.astype(np.int64), list(_C10)


def synthetic_cifar100_test(n_train: int = 2000, n_test: int = 500, seed: int = 0, per_class: int = 20):
    """Near unknowns = blends of two known prototypes; far = unrelated prototypes."""
    rng = np.random.default_rng(seed + 2)
    classes = sorted(set(NEAR_UNKNOWN + FAR_UNKNOWN + ["apple", "baby"]))
    known = _prototypes(10, seed)
    other = _prototypes(len(FAR_UNKNOWN), seed + 99)
    xs, ys = [], []
    for ci, name in enumerate(classes):
        if name in NEAR_UNKNOWN:
            k = NEAR_UNKNOWN.index(name)
            proto = 0.6 * known[k % 10] + 0.4 * known[(k + 3) % 10]
        elif name in FAR_UNKNOWN:
            proto = other[FAR_UNKNOWN.index(name)]
        else:
            proto = other[0]
        xs.append(_render(proto[None], np.zeros(per_class, dtype=int), rng))
        ys.append(np.full(per_class, ci))
    return np.concatenate(xs), np.concatenate(ys).astype(np.int64), classes
