"""Fixed CIFAR-100 *test* classes used only as unknowns during final evaluation.

The grouping is fixed a priori (it must never be revised after seeing results).
Only the official CIFAR-100 **test** partition is ever loaded here; the
CIFAR-100 training images are never touched by this repository.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torchvision

from utils import resolve

NEAR_UNKNOWN = ["bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"]
FAR_UNKNOWN = ["bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"]
GROUPS = {"near": NEAR_UNKNOWN, "far": FAR_UNKNOWN}
IMAGES_PER_GROUP = 800  # 8 classes x 100 test images


def load_cifar100_unknowns(root: str = "data/raw", download: bool = True,
                           synthetic: Optional[dict] = None) -> dict:
    """Return a dict with the unknown evaluation images and their metadata.

    Keys: ``images`` uint8 [1600,32,32,3], ``group`` ('near'/'far'),
    ``fine_label`` (CIFAR-100 fine id), ``fine_name``, ``cifar100_index``
    (index into the official CIFAR-100 test set).
    """
    if synthetic:
        from data.synthetic import synthetic_cifar100_test
        images, fine, classes = synthetic_cifar100_test(**synthetic)
    else:
        ds = torchvision.datasets.CIFAR100(root=str(resolve(root)), train=False, download=download)
        images, fine, classes = np.asarray(ds.data, np.uint8), np.asarray(ds.targets, np.int64), list(ds.classes)

    name_to_id = {n: i for i, n in enumerate(classes)}
    missing = [n for g in GROUPS.values() for n in g if n not in name_to_id]
    if missing:
        raise KeyError(f"CIFAR-100 classes not found: {missing}")

    sel_idx, sel_group = [], []
    for group, names in GROUPS.items():
        ids = [name_to_id[n] for n in names]
        idx = np.flatnonzero(np.isin(fine, ids))
        if not synthetic and len(idx) != IMAGES_PER_GROUP:
            raise RuntimeError(f"Expected {IMAGES_PER_GROUP} {group} images, found {len(idx)}")
        sel_idx.append(idx)
        sel_group += [group] * len(idx)
    sel_idx = np.concatenate(sel_idx)
    return {
        "images": images[sel_idx],
        "group": np.asarray(sel_group),
        "fine_label": fine[sel_idx],
        "fine_name": np.asarray([classes[k] for k in fine[sel_idx]]),
        "cifar100_index": sel_idx.astype(np.int64),
    }
