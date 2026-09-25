"""
PACS protocol shared by Task 2 and Task 3.

* Stratified 80/20 train/val split of each SOURCE domain (seed 6304), built
  once by Task 2 and cached in shared/splits/pacs_sketch_seed6304.json. Task 3
  must reuse that exact file (copy it from the Task 2 results).
* Paths are stored RELATIVE to the PACS root, together with a fingerprint
  (SHA-1 of the sorted relative file list and file sizes). Loading the split
  re-checks the fingerprint against the files on disk, so a different download
  of PACS (renamed or re-encoded files) is detected instead of silently
  producing a different split.
* Domain-balanced batch iterator: 8 images per source domain (+ 24 target
  images for Task 2 adaptation), cycling shorter loaders.
* Seeded DataLoaders (explicit generator + worker seeding) for reproducibility.

Task 3 only ever calls ``load_source_splits`` / ``make_source_loaders``, which
drop the Sketch entry before anything is built.
"""
import hashlib
import json
import os

import numpy as np
from torch.utils.data import DataLoader

from shared.pacs import CLASSES, PACSDomainDataset, eval_transform, list_domain_files, train_transform
from shared.seed import make_generator, worker_init_fn

SOURCE_DOMAINS = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN = "sketch"
SEED = 6304
SPLITS_PATH = os.path.join(os.path.dirname(__file__), "splits", "pacs_sketch_seed6304.json")


def _stratified_split(labels, train_frac, seed):
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in sorted(set(labels)):
        cls_idx = [i for i, l in enumerate(labels) if l == c]
        rng.shuffle(cls_idx)
        n_train = int(round(len(cls_idx) * train_frac))
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return [int(i) for i in train_idx], [int(i) for i in val_idx]


def fingerprint(pacs_root: str, rel_paths: list) -> str:
    h = hashlib.sha1()
    for p in sorted(rel_paths):
        h.update(p.encode())
        h.update(str(os.path.getsize(os.path.join(pacs_root, p))).encode())
    return h.hexdigest()


def build_splits(pacs_root: str, train_frac: float = 0.8, seed: int = SEED, path: str = SPLITS_PATH) -> dict:
    splits = {"seed": seed, "train_frac": train_frac, "format": "relative_paths_v2", "domains": {}}
    for domain in SOURCE_DOMAINS + [TARGET_DOMAIN]:
        fps, labels = list_domain_files(pacs_root, domain)
        rel = [os.path.relpath(p, pacs_root).replace(os.sep, "/") for p in fps]
        entry = {"filepaths": rel, "labels": [int(l) for l in labels],
                 "fingerprint": fingerprint(pacs_root, rel)}
        if domain in SOURCE_DOMAINS:
            entry["train_idx"], entry["val_idx"] = _stratified_split(labels, train_frac, seed)
        splits["domains"][domain] = entry
    splits["fingerprint"] = hashlib.sha1(
        "".join(splits["domains"][d]["fingerprint"] for d in SOURCE_DOMAINS + [TARGET_DOMAIN]).encode()).hexdigest()
    splits["source_fingerprint"] = hashlib.sha1(
        "".join(splits["domains"][d]["fingerprint"] for d in SOURCE_DOMAINS).encode()).hexdigest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(splits, f, indent=1)
    print(f"[splits] built from {pacs_root} -> {path} (source fingerprint {splits['source_fingerprint'][:12]})")
    return splits


def _check(pacs_root, splits, domains):
    for d in domains:
        e = splits["domains"][d]
        missing = [p for p in e["filepaths"][:50] if not os.path.exists(os.path.join(pacs_root, p))]
        if missing:
            raise FileNotFoundError(
                f"Split file lists images that are not under {pacs_root} (e.g. {missing[0]}). "
                f"This PACS copy differs from the one the split was built on. Download PACS with "
                f"shared/download_pacs.py (same source for Task 2 and Task 3).")
        fp = fingerprint(pacs_root, e["filepaths"])
        if fp != e["fingerprint"]:
            raise RuntimeError(f"PACS fingerprint mismatch for domain '{d}': the files on disk are not the "
                               f"ones the split was built on. Use the same PACS download for both tasks.")


def load_splits(pacs_root: str, train_frac: float = 0.8, seed: int = SEED, path: str = SPLITS_PATH,
                allow_build: bool = True) -> dict:
    """Task 2 entry point: load (and verify) the cached split, or build it once."""
    if os.path.exists(path):
        with open(path) as f:
            splits = json.load(f)
        if splits.get("format") != "relative_paths_v2":
            raise RuntimeError(f"{path} is an old-format split (absolute paths from a different PACS copy). "
                               f"Delete it and rebuild with the current code.")
        _check(pacs_root, splits, SOURCE_DOMAINS + [TARGET_DOMAIN])
        return splits
    if not allow_build:
        raise FileNotFoundError(f"{path} not found. Copy the Task 2 split file here.")
    return build_splits(pacs_root, train_frac, seed, path)


def load_source_splits(pacs_root: str, path: str = SPLITS_PATH) -> dict:
    """Task 3 entry point: requires the existing Task 2 split, keeps SOURCE domains only."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Copy Task 2's shared/splits/pacs_sketch_seed6304.json here "
                                f"so Task 3 reuses exactly the same source splits.")
    with open(path) as f:
        splits = json.load(f)
    if splits.get("format") != "relative_paths_v2":
        raise RuntimeError(f"{path} is an old-format split; regenerate it with the updated Task 2 code.")
    splits["domains"] = {d: splits["domains"][d] for d in SOURCE_DOMAINS}  # Sketch entry dropped here
    _check(pacs_root, splits, SOURCE_DOMAINS)
    return splits


def _abs(pacs_root, rel):
    return [os.path.join(pacs_root, p) for p in rel]


def make_source_loaders(splits: dict, pacs_root: str, batch_size_per_domain: int = 8, seed: int = SEED,
                        num_workers: int = 2):
    train_loaders, val_loaders = {}, {}
    for k, domain in enumerate(SOURCE_DOMAINS):
        d = splits["domains"][domain]
        fps = _abs(pacs_root, d["filepaths"])
        train_ds = PACSDomainDataset(fps, d["labels"], d["train_idx"], train_transform())
        val_ds = PACSDomainDataset(fps, d["labels"], d["val_idx"], eval_transform())
        train_loaders[domain] = DataLoader(train_ds, batch_size=batch_size_per_domain, shuffle=True,
                                           num_workers=num_workers, drop_last=True, worker_init_fn=worker_init_fn,
                                           generator=make_generator(seed + k), persistent_workers=num_workers > 0)
        val_loaders[domain] = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    return train_loaders, val_loaders


def source_val_datasets(splits: dict, pacs_root: str) -> dict:
    out = {}
    for domain in SOURCE_DOMAINS:
        d = splits["domains"][domain]
        out[domain] = PACSDomainDataset(_abs(pacs_root, d["filepaths"]), d["labels"], d["val_idx"], eval_transform())
    return out


def make_target_loaders(splits: dict, pacs_root: str, target_batch_size: int = 24, seed: int = SEED,
                        num_workers: int = 2):
    """Task 2 only. The training loader's labels are never used by any method."""
    t = splits["domains"][TARGET_DOMAIN]
    fps = _abs(pacs_root, t["filepaths"])
    all_idx = list(range(len(fps)))
    train_ds = PACSDomainDataset(fps, t["labels"], all_idx, train_transform())
    eval_ds = PACSDomainDataset(fps, t["labels"], all_idx, eval_transform())
    target_loader = DataLoader(train_ds, batch_size=target_batch_size, shuffle=True, num_workers=num_workers,
                               drop_last=True, worker_init_fn=worker_init_fn, generator=make_generator(seed + 99),
                               persistent_workers=num_workers > 0)
    target_eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    return target_loader, target_eval_loader


class _CyclingIter:
    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


class DomainBalancedBatchIterator:
    """Yields {"source": {domain: (x, y)}, "target": (x, y_UNUSED) or None} forever."""

    def __init__(self, train_loaders: dict, target_loader=None):
        self._source_iters = {d: _CyclingIter(l) for d, l in train_loaders.items()}
        self._target_iter = _CyclingIter(target_loader) if target_loader is not None else None

    def __iter__(self):
        return self

    def __next__(self):
        source_batch = {d: next(it) for d, it in self._source_iters.items()}
        target_batch = next(self._target_iter) if self._target_iter is not None else None
        return {"source": source_batch, "target": target_batch}


__all__ = ["SOURCE_DOMAINS", "TARGET_DOMAIN", "CLASSES", "load_splits", "load_source_splits", "build_splits",
           "make_source_loaders", "make_target_loaders", "source_val_datasets", "DomainBalancedBatchIterator"]
