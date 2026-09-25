"""
Single shared training loop for Source-only, DAN, DANN, and CDAN. Method
files under methods/ supply only compute_step(); everything else (batching,
BatchNorm freezing, optimizer, validation, early stopping, checkpointing) is
common, per the assignment's suggested repo structure.

Usage:
    python train.py --config configs/source_only.yaml
    python train.py --config configs/dan.yaml
    python train.py --config configs/dan.yaml --override lambda_mmd=10   # controlled study
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # task2/

from shared.pacs_protocol import build_or_load_splits, make_loaders, DomainBalancedBatchIterator, SOURCE_DOMAINS
from models.backbone import ResNet18Backbone, freeze_batchnorm_running_stats, FEAT_DIM
from models.classifier_head import ClassifierHead, NUM_CLASSES
from models.domain_discriminator import DomainDiscriminator
from methods.source_only import SourceOnlyMethod
from methods.dan import DANMethod
from methods.dann import DANNMethod
from methods.cdan import CDANMethod
from evaluation.metrics import domain_metrics, mean_across_domains


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str, overrides: dict = None) -> dict:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        method_cfg = yaml.safe_load(f)
    base_path = os.path.join(config_dir, method_cfg["base_config"])
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in method_cfg.items() if k != "base_config"})
    if overrides:
        cfg.update(overrides)
    return cfg


def build_method(cfg: dict, backbone, head, device):
    method_name = cfg["method"]
    if method_name == "source_only":
        return SourceOnlyMethod(backbone, head, device)
    if method_name == "dan":
        return DANMethod(backbone, head, device, lambda_mmd=cfg.get("lambda_mmd", 1.0))
    if method_name == "dann":
        disc = DomainDiscriminator(FEAT_DIM, cfg["discriminator"]["hidden_dim"]).to(device)
        return DANNMethod(backbone, head, device, disc, max_alpha=cfg.get("max_alpha", 1.0))
    if method_name == "cdan":
        disc = DomainDiscriminator(FEAT_DIM * NUM_CLASSES, cfg["discriminator"]["hidden_dim"]).to(device)
        return CDANMethod(backbone, head, device, disc, max_alpha=cfg.get("max_alpha", 1.0))
    raise ValueError(f"Unknown method: {method_name}")


@torch.no_grad()
def evaluate_on_loader(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        feat = backbone(x)
        logits = head(feat)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    return domain_metrics(y_true, y_pred)


def validate_on_sources(backbone, head, val_loaders: dict, device) -> dict:
    per_domain = {d: evaluate_on_loader(backbone, head, loader, device)
                  for d, loader in val_loaders.items()}
    mean = mean_across_domains(per_domain)
    return {"per_domain": per_domain, "mean": mean}


def run_training(cfg: dict, run_name_suffix: str = ""):
    seed = cfg["seed"]
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{cfg['run_name']}{run_name_suffix}] device={device}")

    splits = build_or_load_splits(cfg["pacs"]["root"], cfg["pacs"]["train_val_split"], seed)
    train_loaders, val_loaders, target_loader, target_eval_loader = make_loaders(
        splits, batch_size_per_domain=cfg["batch"]["per_source_domain"],
        target_batch_size=cfg["batch"]["target"])

    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(FEAT_DIM, NUM_CLASSES).to(device)
    method = build_method(cfg, backbone, head, device)

    needs_target = cfg["method"] != "source_only"
    batch_iter = DomainBalancedBatchIterator(
        train_loaders, target_loader=target_loader if needs_target else None)

    params = list(backbone.parameters()) + list(head.parameters()) + list(method.extra_parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])

    steps_per_epoch = max(len(l) for l in train_loaders.values())
    max_epochs = cfg["optim"]["max_epochs"]
    total_steps = steps_per_epoch * max_epochs

    best_mean_f1 = -1.0
    epochs_without_improve = 0
    best_state = None
    history = []
    global_step = 0

    results_dir = cfg["paths"]["results_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    run_name = cfg["run_name"] + run_name_suffix

    for epoch in range(max_epochs):
        backbone.train()
        head.train()
        if hasattr(method, "discriminator"):
            method.discriminator.train()
        freeze_batchnorm_running_stats(backbone)  # must be called AFTER .train()

        epoch_logs = []
        for step in range(steps_per_epoch):
            batch = next(batch_iter)
            source_batches = batch["source"]
            target_batch = batch["target"]
            progress = global_step / max(1, total_steps)

            optimizer.zero_grad()
            out = method.compute_step(source_batches, target_batch, progress)
            out["loss"].backward()
            optimizer.step()

            epoch_logs.append({k: v for k, v in out.items() if k != "loss"} | {"loss": out["loss"].item()})
            global_step += 1

        val_report = validate_on_sources(backbone, head, val_loaders, device)
        mean_f1 = val_report["mean"]["macro_f1"]
        print(f"[{run_name}] epoch {epoch}: mean source-val macro-F1 = {mean_f1:.4f} "
              f"(per-domain: { {d: round(m['macro_f1'], 3) for d, m in val_report['per_domain'].items()} })")

        history.append({"epoch": epoch, "val": val_report,
                         "mean_train_loss": float(np.mean([l["loss"] for l in epoch_logs])),
                         "step_logs_sample": epoch_logs[-1]})  # last step's logs as a lightweight sample

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            epochs_without_improve = 0
            best_state = {
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            if hasattr(method, "discriminator"):
                best_state["discriminator"] = {k: v.cpu().clone()
                                                for k, v in method.discriminator.state_dict().items()}
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg["optim"]["early_stop_patience"]:
                print(f"[{run_name}] early stopping at epoch {epoch} "
                      f"(best mean macro-F1 = {best_mean_f1:.4f})")
                break

    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    torch.save({"state": best_state, "cfg": cfg, "best_mean_source_val_macro_f1": best_mean_f1},
               ckpt_path)
    history_path = os.path.join(results_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[{run_name}] done. best mean source-val macro-F1 = {best_mean_f1:.4f}")
    print(f"[{run_name}] checkpoint: {ckpt_path}")
    print(f"[{run_name}] history:    {history_path}")
    return ckpt_path, history_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[],
                         help="key=value, e.g. --override lambda_mmd=10")
    args = parser.parse_args()

    overrides = {}
    for kv in args.override:
        k, v = kv.split("=", 1)
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        overrides[k] = v

    cfg = load_config(args.config, overrides)
    suffix = ""
    if overrides:
        suffix = "_" + "_".join(f"{k}{v}" for k, v in overrides.items())
    run_training(cfg, run_name_suffix=suffix)


if __name__ == "__main__":
    main()
