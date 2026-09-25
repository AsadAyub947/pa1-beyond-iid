"""CIFAR-10 (known classes): raw arrays, transforms and split-aware datasets.

Every split is built from the *official* CIFAR-10 arrays plus the fixed index
file written by ``data/make_splits.py``:

* ``train`` – 90 % of the official training partition (optimisation only)
* ``val``   – 10 % of the official training partition (checkpoint selection,
              rejection-threshold calibration)
* ``test``  – the complete official CIFAR-10 test set (final known evaluation)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

from utils import resolve

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------- #
# Raw arrays
# --------------------------------------------------------------------------- #
def load_cifar10_arrays(root: str, train: bool, download: bool = True,
                        synthetic: Optional[dict] = None):
    """Return ``(images uint8 [N,32,32,3], labels int64 [N], class_names)``."""
    if synthetic:
        from data.synthetic import synthetic_cifar10
        return synthetic_cifar10(train=train, **synthetic)
    ds = torchvision.datasets.CIFAR10(root=str(resolve(root)), train=train, download=download)
    return np.asarray(ds.data, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64), list(ds.classes)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def build_transform(train: bool, randaugment: Optional[dict] = None) -> Callable:
    """Training: RandomCrop(32, pad 4) -> HFlip -> [RandAugment] -> ToTensor -> Normalize.

    Evaluation / feature extraction: ToTensor -> Normalize (no augmentation).
    RandAugment (GCSC only) is inserted *after* crop+flip and *before*
    conversion/normalisation, exactly as required.
    """
    ops = []
    if train:
        ops += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
        if randaugment:
            ops.append(T.RandAugment(num_ops=int(randaugment.get("num_ops", 2)),
                                     magnitude=int(randaugment.get("magnitude", 9))))
    ops += [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return T.Compose(ops)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class ArrayDataset(Dataset):
    """Images held as a uint8 array; returns ``(tensor, label, index)``."""

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 indices: Optional[np.ndarray] = None, transform: Optional[Callable] = None):
        from PIL import Image  # local import keeps module import cheap
        self._Image = Image
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(images)) if indices is None else np.asarray(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = self._Image.fromarray(self.images[idx])
        x = self.transform(img) if self.transform is not None else img
        return x, int(self.labels[idx]), idx


def load_split_indices(split_file: str) -> dict:
    p = resolve(split_file)
    if not p.exists():
        raise FileNotFoundError(
            f"Split file {p} not found. Run `python -m data.make_splits` first.")
    with open(p) as f:
        s = json.load(f)
    return {"train": np.asarray(s["train_idx"], dtype=np.int64),
            "val": np.asarray(s["val_idx"], dtype=np.int64),
            "meta": s}


def build_cifar10_datasets(data_cfg: dict, train_transform: Optional[Callable] = None,
                           eval_transform: Optional[Callable] = None,
                           augment_train: bool = True) -> dict:
    """Return a dict with ``train``, ``train_eval`` (unaugmented), ``val`` and ``test`` datasets."""
    root = data_cfg.get("root", "data/raw")
    synthetic = data_cfg.get("synthetic")
    download = bool(data_cfg.get("download", True))
    tr_x, tr_y, classes = load_cifar10_arrays(root, train=True, download=download, synthetic=synthetic)
    te_x, te_y, _ = load_cifar10_arrays(root, train=False, download=download, synthetic=synthetic)
    split = load_split_indices(data_cfg["split_file"])
    if len(split["train"]) + len(split["val"]) != len(tr_x):
        raise RuntimeError("Split file does not match the CIFAR-10 training partition size; "
                           "re-run `python -m data.make_splits` with the same data settings.")

    eval_tf = eval_transform or build_transform(train=False)
    train_tf = train_transform or (build_transform(train=True) if augment_train else eval_tf)
    return {
        "train": ArrayDataset(tr_x, tr_y, split["train"], train_tf),
        "train_eval": ArrayDataset(tr_x, tr_y, split["train"], eval_tf),
        "val": ArrayDataset(tr_x, tr_y, split["val"], eval_tf),
        "test": ArrayDataset(te_x, te_y, None, eval_tf),
        "classes": classes,
    }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int,
                generator: Optional[torch.Generator] = None, drop_last: bool = False):
    from utils import seed_worker
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), drop_last=drop_last,
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=False)
