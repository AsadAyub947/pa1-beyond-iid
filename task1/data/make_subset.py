"""
Build:
  1. A stratified 80/20 train/val split of the official STL-10 train partition
     (seed 6304), used to train each backbone's linear classifier head.
  2. A class-balanced subset of 500 images from the official STL-10 test
     partition (seed 6304), used for every subsequent evaluation/intervention.

Both selections are SAVED (image indices, not re-sampled implicitly elsewhere)
so that every model and every intervention operates on the exact same images.
"""
import argparse
import json
import os

import numpy as np
import yaml
from torchvision.datasets import STL10


def stratified_train_val_split(labels: np.ndarray, train_frac: float, seed: int):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        n_train = int(round(len(cls_idx) * train_frac))
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def class_balanced_subset(labels: np.ndarray, n_total: int, seed: int, n_classes: int):
    rng = np.random.RandomState(seed)
    per_class = n_total // n_classes
    selected = []
    imbalance_report = {}
    for c in range(n_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)
        take = min(per_class, len(cls_idx))
        if take < per_class:
            imbalance_report[int(c)] = {"requested": per_class, "available": int(len(cls_idx))}
        selected.extend(cls_idx[:take].tolist())
    rng.shuffle(selected)
    return selected, imbalance_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    data_root = cfg["dataset"]["root"]
    n_classes = cfg["dataset"]["num_classes"]
    n_test = cfg["subset"]["n_test"]
    train_frac = cfg["subset"]["train_val_split"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("Downloading / loading STL-10 (train + test official partitions)...")
    train_ds = STL10(root=data_root, split="train", download=True)
    test_ds = STL10(root=data_root, split="test", download=True)

    train_labels = np.array(train_ds.labels)
    test_labels = np.array(test_ds.labels)

    train_idx, val_idx = stratified_train_val_split(train_labels, train_frac, seed)
    test_subset_idx, imbalance = class_balanced_subset(test_labels, n_test, seed, n_classes)

    out = {
        "seed": seed,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_subset_idx": test_subset_idx,
        "test_subset_size": len(test_subset_idx),
        "class_imbalance_in_test_subset": imbalance,
    }
    out_path = os.path.join(results_dir, "subset_indices.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Train split: {len(train_idx)} images | Val split: {len(val_idx)} images")
    print(f"Class-balanced test subset: {len(test_subset_idx)} images")
    if imbalance:
        print("WARNING - class imbalance in test subset (fewer than requested "
              f"per-class examples available): {imbalance}")
    print(f"Saved indices to {out_path}")


if __name__ == "__main__":
    main()
