"""
Training loop for DAN-DG and SAM. ERM is intentionally NOT trainable through
this script -- see methods/erm.py and configs/erm.yaml.

Usage:
    python train.py --config configs/dan_dg.yaml
    python train.py --config configs/sam.yaml
    python train.py --config configs/dan_dg.yaml --override lambda_dg=10   # controlled study
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (for shared/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                    # task3/

from shared.pacs_protocol import build_or_load_splits, make_loaders, DomainBalancedBatchIterator
from models.backbone import ResNet18Backbone, freeze_batchnorm_running_stats, FEAT_DIM
from models.classifier_head import ClassifierHead, NUM_CLASSES
from methods.dan_dg import DANDGMethod
from methods.sam import SAMMethod
from selection.source_validation import evaluate_on_source_val, EarlyStopTracker


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_path: str, overrides: dict = None) -> dict:
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        method_cfg = yaml.safe_load(f)
    if method_cfg["method"] == "erm":
        raise SystemExit(
            "configs/erm.yaml cannot be passed to train.py: ERM is not retrained in "
            "Task 3 -- it must be the exact checkpoint from Task 2's Source-only run. "
            "See evaluate_sketch.py, which loads it via methods/erm.py::load_erm_checkpoint.")
    base_path = os.path.join(config_dir, method_cfg["base_config"])
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in method_cfg.items() if k != "base_config"})
    if overrides:
        cfg.update(overrides)
    return cfg


def build_method(cfg: dict, backbone, head, device):
    if cfg["method"] == "dan_dg":
        return DANDGMethod(backbone, head, device, lambda_dg=cfg.get("lambda_dg", 1.0))
    if cfg["method"] == "sam":
        return SAMMethod(backbone, head, device, rho=cfg.get("rho", 0.05))
    raise ValueError(f"Unknown trainable method: {cfg['method']}")


def run_training(cfg: dict, run_name_suffix: str = ""):
    seed = cfg["seed"]
    set_all_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg["run_name"] + run_name_suffix
    print(f"[{run_name}] device={device}")

    splits = build_or_load_splits(cfg["pacs"]["root"], cfg["pacs"]["train_val_split"], seed)
    # Task 3 never touches the target loader at all -- request only source loaders.
    train_loaders, val_loaders, _target_loader_UNUSED, _target_eval_UNUSED = make_loaders(
        splits, batch_size_per_domain=cfg["batch"]["per_source_domain"], target_batch_size=1)

    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(FEAT_DIM, NUM_CLASSES).to(device)
    method = build_method(cfg, backbone, head, device)

    batch_iter = DomainBalancedBatchIterator(train_loaders, target_loader=None)

    all_params = list(backbone.parameters()) + list(head.parameters()) + list(method.extra_parameters())
    if cfg["method"] == "sam":
        optimizer = method.make_optimizer(all_params, lr=cfg["optim"]["lr"],
                                           weight_decay=cfg["optim"]["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(all_params, lr=cfg["optim"]["lr"],
                                       weight_decay=cfg["optim"]["weight_decay"])

    steps_per_epoch = max(len(l) for l in train_loaders.values())
    max_epochs = cfg["optim"]["max_epochs"]
    tracker = EarlyStopTracker(patience=cfg["optim"]["early_stop_patience"])
    history = []

    results_dir = cfg["paths"]["results_dir"]
    ckpt_dir = cfg["paths"]["checkpoints_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(max_epochs):
        backbone.train()
        head.train()
        freeze_batchnorm_running_stats(backbone)  # frozen across BOTH SAM passes too

        epoch_logs = []
        for _ in range(steps_per_epoch):
            batch = next(batch_iter)
            source_batches = batch["source"]
            logs = method.step(optimizer, source_batches)
            epoch_logs.append(logs)

        val_report = evaluate_on_source_val(backbone, head, val_loaders, device)
        mean_f1 = val_report["mean"]["macro_f1"]
        print(f"[{run_name}] epoch {epoch}: mean source-val macro-F1 = {mean_f1:.4f} "
              f"| mean train loss = {np.mean([l['loss'] for l in epoch_logs]):.4f}")

        history.append({"epoch": epoch, "val": val_report,
                         "mean_train_loss": float(np.mean([l["loss"] for l in epoch_logs])),
                         "step_logs_sample": epoch_logs[-1]})

        def snapshot_state():
            state = {
                "backbone": {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            return state

        should_stop = tracker.update(mean_f1, snapshot_state)
        if should_stop:
            print(f"[{run_name}] early stopping at epoch {epoch} "
                  f"(best mean macro-F1 = {tracker.best_mean_f1:.4f})")
            break

    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")
    torch.save({"state": tracker.best_state, "cfg": cfg,
                "best_mean_source_val_macro_f1": tracker.best_mean_f1}, ckpt_path)
    history_path = os.path.join(results_dir, f"{run_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[{run_name}] done. best mean source-val macro-F1 = {tracker.best_mean_f1:.4f}")
    print(f"[{run_name}] checkpoint: {ckpt_path}")
    return ckpt_path, history_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
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
    suffix = "_" + "_".join(f"{k}{v}" for k, v in overrides.items()) if overrides else ""
    run_training(cfg, run_name_suffix=suffix)


if __name__ == "__main__":
    main()
