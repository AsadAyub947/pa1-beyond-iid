"""Train one model (Vanilla / GCSC / PROSER / RPL) on the CIFAR-10 90 % split.

    python train.py --config configs/vanilla.yaml
    python train.py --config configs/gcsc.yaml
    python train.py --config configs/proser.yaml      # needs checkpoints/vanilla/best.pt
    python train.py --config configs/rpl.yaml         # optional
    python train.py --config configs/vanilla.yaml --resume   # continue an interrupted run

Only the CIFAR-10 *training* portion is used for optimisation; after every
epoch the model is evaluated on the CIFAR-10 *validation* portion and the
checkpoint with the highest validation accuracy is saved as ``best.pt``.
No CIFAR-100 data is loaded by this script.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import time
from collections import defaultdict

import numpy as np
import torch

from data.cifar10 import build_cifar10_datasets, make_loader
from methods import get_method
from utils import dump_json, get_device, load_config, resolve, set_seed


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate_accuracy(model, loader, method, device) -> float:
    model.eval()
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = method.known_logits(model(x)).float()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[], help="e.g. train.epochs=1 data.num_workers=0")
    ap.add_argument("--resume", action="store_true", help="resume from <checkpoint_dir>/last.pt")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config), args.override)
    tcfg, dcfg = cfg["train"], cfg["data"]
    seed = int(cfg.get("seed", 6304))
    set_seed(seed, deterministic=bool(tcfg.get("deterministic", False)))
    device = get_device(cfg.get("device", "auto"))
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    ckpt_dir = resolve(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    method = get_method(cfg)
    ds = build_cifar10_datasets(dcfg, train_transform=method.train_transform())
    nw = int(dcfg.get("num_workers", 4))
    gen = torch.Generator()
    train_loader = make_loader(ds["train"], int(tcfg["batch_size"]), shuffle=True, num_workers=nw, generator=gen)
    val_loader = make_loader(ds["val"], int(tcfg.get("eval_batch_size", 512)), shuffle=False, num_workers=nw)
    print(f"[train] {cfg['name']}: train={len(ds['train'])} val={len(ds['val'])} device={device} amp={use_amp}")
    print(f"[train] train transform: {method.train_transform()}")

    model = method.build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(tcfg["lr"]), momentum=float(tcfg["momentum"]),
                                weight_decay=float(tcfg["weight_decay"]), nesterov=bool(tcfg.get("nesterov", False)))
    epochs = int(tcfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = make_scaler(use_amp)

    start_epoch, best_acc, best_epoch, history = 0, -1.0, -1, []
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if args.resume and last_path.exists():
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        set_rng_state(st["rng"])
        start_epoch, best_acc, best_epoch, history = st["epoch"], st["best_acc"], st["best_epoch"], st["history"]
        print(f"[train] resumed from {last_path} at epoch {start_epoch} (best val acc {best_acc:.4f})")

    for epoch in range(start_epoch, epochs):
        gen.manual_seed(seed * 1000 + epoch)  # reproducible shuffling/augmentation per epoch
        model.train()
        t0 = time.time()
        agg = defaultdict(float)
        n_batches = 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, stats = method.training_loss(model, x, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch + 1}")
            agg["loss"] += loss.item()
            for k, v in stats.items():
                if k in ("correct", "n"):
                    agg[k] += v
                elif v == v:  # skip NaN
                    agg[k] += v
            n_batches += 1
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        val_acc = evaluate_accuracy(model, val_loader, method, device)
        row = {"epoch": epoch + 1, "lr": lr_now, "train_loss": agg["loss"] / max(n_batches, 1),
               "train_acc": agg["correct"] / max(agg["n"], 1), "val_acc": val_acc,
               "time_s": time.time() - t0}
        for k, v in agg.items():
            if k not in ("loss", "correct", "n"):
                row[k] = v / max(n_batches, 1)
        history.append(row)

        improved = val_acc > best_acc
        if improved:
            best_acc, best_epoch = val_acc, epoch + 1
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val_acc": val_acc,
                        "config": cfg, "method": cfg["method"]}, best_path)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "rng": rng_state(), "epoch": epoch + 1, "best_acc": best_acc,
                    "best_epoch": best_epoch, "history": history}, last_path)
        extra = " ".join(f"{k}={v:.4f}" for k, v in row.items()
                         if k not in ("epoch", "lr", "train_loss", "train_acc", "val_acc", "time_s"))
        print(f"[{cfg['name']}] ep {epoch + 1:3d}/{epochs} lr {lr_now:.5f} loss {row['train_loss']:.4f} "
              f"train_acc {row['train_acc']:.4f} val_acc {val_acc:.4f}{' *' if improved else ''} "
              f"({row['time_s']:.0f}s) {extra}", flush=True)

    # ---- logs -----------------------------------------------------------------
    if history:
        keys = list(dict.fromkeys(k for r in history for k in r))
        with open(ckpt_dir / "train_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)
    dump_json({"name": cfg["name"], "method": cfg["method"], "best_epoch": best_epoch,
               "best_val_acc": best_acc, "epochs": epochs, "config": cfg}, ckpt_dir / "train_summary.json")
    print(f"[train] done. best val acc {best_acc:.4f} at epoch {best_epoch} -> {best_path}")


if __name__ == "__main__":
    main()
