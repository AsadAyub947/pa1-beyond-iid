"""Create the stratified 90/10 split of the official CIFAR-10 training partition.

Usage (from the task4/ directory):
    python -m data.make_splits                     # downloads CIFAR-10 + CIFAR-100 test
    python -m data.make_splits --config configs/vanilla.yaml

The split uses seed 6304 and is written once to ``data/splits/``; every
method reads the same index file, so all models see identical train/val sets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python data/make_splits.py`

from sklearn.model_selection import train_test_split  # noqa: E402

from data.cifar10 import load_cifar10_arrays  # noqa: E402
from data.cifar100_unknowns import load_cifar100_unknowns  # noqa: E402
from utils import dump_json, load_config, resolve  # noqa: E402


def make_split(labels: np.ndarray, val_fraction: float = 0.1, seed: int = 6304):
    idx = np.arange(len(labels))
    tr, va = train_test_split(idx, test_size=val_fraction, stratify=labels,
                              random_state=seed, shuffle=True)
    return np.sort(tr), np.sort(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vanilla.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--force", action="store_true", help="overwrite an existing split file")
    args = ap.parse_args()
    cfg = load_config(resolve(args.config), args.override)
    dcfg = cfg["data"]
    seed = int(cfg.get("seed", 6304))
    out = resolve(dcfg["split_file"])
    if out.exists() and not args.force:
        print(f"[make_splits] {out} already exists – keeping it (use --force to regenerate).")
        return

    _, y, classes = load_cifar10_arrays(dcfg.get("root", "data/raw"), train=True,
                                        download=dcfg.get("download", True),
                                        synthetic=dcfg.get("synthetic"))
    tr, va = make_split(y, float(dcfg.get("val_fraction", 0.1)), seed)
    counts_tr = np.bincount(y[tr], minlength=len(classes)).tolist()
    counts_va = np.bincount(y[va], minlength=len(classes)).tolist()
    dump_json({"seed": seed, "val_fraction": dcfg.get("val_fraction", 0.1),
               "n_train": len(tr), "n_val": len(va),
               "train_class_counts": counts_tr, "val_class_counts": counts_va,
               "classes": classes, "train_idx": tr.tolist(), "val_idx": va.tolist()}, out)
    print(f"[make_splits] train={len(tr)} val={len(va)} -> {out}")
    print(f"[make_splits] per-class train counts {counts_tr}")
    print(f"[make_splits] per-class val counts   {counts_va}")

    # Download/verify the CIFAR-100 *test* partition now so later steps work offline.
    # Only metadata sizes are printed; no unknown image is used for anything here.
    unk = load_cifar100_unknowns(dcfg.get("root", "data/raw"), download=dcfg.get("download", True),
                                 synthetic=dcfg.get("synthetic"))
    print(f"[make_splits] CIFAR-100 unknown pool ready: "
          f"near={int((unk['group'] == 'near').sum())} far={int((unk['group'] == 'far').sum())}")


if __name__ == "__main__":
    main()
