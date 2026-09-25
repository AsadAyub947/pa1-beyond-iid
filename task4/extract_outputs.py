"""Freeze a trained model and cache its penultimate features f(x) and logits z(x).

    python extract_outputs.py --config configs/vanilla.yaml configs/gcsc.yaml configs/proser.yaml
    python extract_outputs.py --config configs/rpl.yaml            # optional model

For each model this writes (float32, no augmentation, eval mode):
    cache/<name>/known.npz    CIFAR-10 train (90 %, unaugmented), val (10 %), test
    cache/<name>/unknown.npz  the fixed CIFAR-100 near/far test images
    cache/<name>/meta.json    checkpoint hash, epoch and validation accuracy

Every score (MSP, MLS, Energy, Mahalanobis, PROSER, RPL) is later computed from
these *same* saved arrays. Unknown outputs are only extracted once all
required checkpoints exist (protocol: unknowns are evaluated only after every
checkpoint is fixed); the checkpoint hashes are then frozen in
``results/frozen_manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from data.cifar10 import ArrayDataset, build_cifar10_datasets, build_transform, make_loader
from data.cifar100_unknowns import load_cifar100_unknowns
from methods import get_method
from utils import dump_json, get_device, load_config, resolve, sha256_file

REQUIRED_CHECKPOINTS = ["checkpoints/vanilla/best.pt", "checkpoints/gcsc/best.pt", "checkpoints/proser/best.pt"]
MANIFEST = "results/frozen_manifest.json"


@torch.no_grad()
def run_inference(model, method, loader, device) -> dict:
    model.eval()
    feats, logits, extras, labels, index = [], [], {}, [], []
    for x, y, idx in loader:
        out = model(x.to(device, non_blocking=True))
        feats.append(out["features"].float().cpu())
        logits.append(method.known_logits(out).float().cpu())
        for k, v in method.extra_outputs(out).items():
            extras.setdefault(k, []).append(v.float().cpu())
        labels.append(y)
        index.append(idx)
    res = {"features": torch.cat(feats).numpy(), "logits": torch.cat(logits).numpy(),
           "labels": torch.cat(labels).numpy(), "index": torch.cat(index).numpy()}
    for k, v in extras.items():
        res[k] = torch.cat(v).numpy()
    return res


def freeze_check(skip: bool) -> dict:
    """Refuse to touch unknowns before all required checkpoints exist; record hashes."""
    missing = [p for p in REQUIRED_CHECKPOINTS if not resolve(p).exists()]
    if missing and not skip:
        raise SystemExit("Refusing to extract CIFAR-100 unknown outputs: these checkpoints are not "
                         f"fixed yet: {missing}. Finish training first (or pass --skip-freeze-check "
                         "for debugging only).")
    hashes = {p: sha256_file(resolve(p)) for p in REQUIRED_CHECKPOINTS if resolve(p).exists()}
    rpl = resolve("checkpoints/rpl/best.pt")
    if rpl.exists():
        hashes["checkpoints/rpl/best.pt"] = sha256_file(rpl)
    man_path = resolve(MANIFEST)
    if man_path.exists():
        old = json.load(open(man_path))
        changed = [p for p, h in old.get("checkpoints", {}).items() if p in hashes and hashes[p] != h]
        if changed and not skip:
            raise SystemExit(f"Checkpoint(s) {changed} changed after unknown outputs were first "
                             f"extracted (see {MANIFEST}). Revising models after seeing unknowns "
                             "violates the protocol.")
        hashes = {**old.get("checkpoints", {}), **hashes}
        first = old.get("first_unknown_extraction")
    else:
        first = time.strftime("%Y-%m-%d %H:%M:%S")
    dump_json({"first_unknown_extraction": first, "checkpoints": hashes,
               "note": "Checkpoints and score definitions frozen before CIFAR-100 unknowns were evaluated."},
              man_path)
    return hashes


def extract_one(cfg: dict, what: str, skip_freeze: bool) -> None:
    device = get_device(cfg.get("device", "auto"))
    method = get_method(cfg)
    ckpt_path = resolve(cfg["output"]["checkpoint_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} missing – train this model first.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = method.build_model_skeleton()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = resolve(cfg["output"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dcfg = cfg["data"]
    bs = int(cfg["train"].get("eval_batch_size", 512))
    nw = int(dcfg.get("num_workers", 4))

    if what in ("known", "all"):
        ds = build_cifar10_datasets(dcfg, augment_train=False)
        arrays = {}
        for split, key in (("train", "train_eval"), ("val", "val"), ("test", "test")):
            res = run_inference(model, method, make_loader(ds[key], bs, False, nw), device)
            for k, v in res.items():
                arrays[f"{split}_{k}"] = v
            acc = float((res["logits"].argmax(1) == res["labels"]).mean())
            print(f"[extract] {cfg['name']} {split:5s}: n={len(res['labels'])} acc={acc:.4f}")
        np.savez_compressed(out_dir / "known.npz", **arrays)

    if what in ("unknown", "all"):
        freeze_check(skip_freeze)
        unk = load_cifar100_unknowns(dcfg.get("root", "data/raw"), download=dcfg.get("download", True),
                                     synthetic=dcfg.get("synthetic"))
        uds = ArrayDataset(unk["images"], np.zeros(len(unk["images"]), dtype=np.int64), None,
                           build_transform(train=False))
        res = run_inference(model, method, make_loader(uds, bs, False, nw), device)
        res.pop("labels")
        res.pop("index")
        np.savez_compressed(out_dir / "unknown.npz", group=unk["group"], fine_name=unk["fine_name"],
                            fine_label=unk["fine_label"], cifar100_index=unk["cifar100_index"], **res)
        print(f"[extract] {cfg['name']} unknown: near={int((unk['group'] == 'near').sum())} "
              f"far={int((unk['group'] == 'far').sum())}")

    dump_json({"name": cfg["name"], "method": cfg["method"], "checkpoint": str(ckpt_path),
               "checkpoint_sha256": sha256_file(ckpt_path), "epoch": ckpt.get("epoch"),
               "val_acc_at_selection": ckpt.get("val_acc"), "extracted": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config_path": cfg.get("_config_path")}, out_dir / "meta.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    ap.add_argument("--what", choices=["known", "unknown", "all"], default="all")
    ap.add_argument("--skip-freeze-check", action="store_true", help="debugging only")
    args = ap.parse_args()
    for c in args.config:
        extract_one(load_config(resolve(c), args.override), args.what, args.skip_freeze_check)


if __name__ == "__main__":
    main()
