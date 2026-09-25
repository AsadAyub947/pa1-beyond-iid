"""
Final evaluation stage. Run this ONLY after all four method checkpoints exist
(i.e., after train.py has been run for source_only, dan, dann, cdan) --
target labels are used here for the first time, exactly as the assignment's
transductive protocol requires ("evaluated using its labels only after all
models, settings, and checkpoints have been fixed").

Usage:
    python evaluate_final.py --base_config configs/base.yaml
    python evaluate_final.py --base_config configs/base.yaml --controlled_study dan
    python evaluate_final.py --base_config configs/base.yaml --controlled_study dann
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.pacs_protocol import build_or_load_splits, make_loaders, SOURCE_DOMAINS, TARGET_DOMAIN
from shared.pacs import CLASSES
from models.backbone import ResNet18Backbone, FEAT_DIM
from models.classifier_head import ClassifierHead, NUM_CLASSES
from evaluation.metrics import domain_metrics, mean_across_domains
from evaluation.domain_separability import domain_separability_score
from evaluation.class_analysis import per_class_accuracy, class_deltas_vs_source_only, confusion_for_classes
from train import load_config, run_training


@torch.no_grad()
def get_preds_and_features(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    all_preds, all_labels, all_feats = [], [], []
    for x, y in loader:
        x = x.to(device)
        feat = backbone(x)
        logits = head(feat)
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
        all_feats.append(feat.cpu().numpy())
    return (np.concatenate(all_preds), np.concatenate(all_labels),
            np.concatenate(all_feats, axis=0))


def load_method_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(FEAT_DIM, NUM_CLASSES).to(device)
    backbone.load_state_dict(ckpt["state"]["backbone"])
    head.load_state_dict(ckpt["state"]["head"])
    return backbone, head, ckpt


def evaluate_method(run_name: str, ckpt_path: str, val_loaders: dict, target_eval_loader,
                     device) -> dict:
    backbone, head, ckpt = load_method_model(ckpt_path, device)

    per_domain = {}
    val_feats_pooled = []
    for domain, loader in val_loaders.items():
        preds, labels, feats = get_preds_and_features(backbone, head, loader, device)
        per_domain[domain] = domain_metrics(labels, preds)
        val_feats_pooled.append(feats)
    val_feats_pooled = np.concatenate(val_feats_pooled, axis=0)
    mean_source = mean_across_domains(per_domain)

    target_preds, target_labels, target_feats = get_preds_and_features(
        backbone, head, target_eval_loader, device)
    target_report = domain_metrics(target_labels, target_preds)

    sep = domain_separability_score(val_feats_pooled, target_feats)

    per_class_acc = per_class_accuracy(target_labels, target_preds, NUM_CLASSES)

    return {
        "run_name": run_name,
        "source_val_per_domain": per_domain,
        "source_val_mean": mean_source,
        "target": target_report,
        "domain_separability": sep,
        "per_class_target_accuracy": {CLASSES[i]: (None if np.isnan(per_class_acc[i]) else float(per_class_acc[i]))
                                       for i in range(NUM_CLASSES)},
        "_target_preds": target_preds,       # kept for confusion analysis; stripped before final JSON dump
        "_target_labels": target_labels,
        "_per_class_acc_array": per_class_acc,
        "best_mean_source_val_macro_f1_at_selection": ckpt["best_mean_source_val_macro_f1"],
    }


def build_comparison_table(reports: dict) -> list:
    rows = []
    source_only_target_acc = reports["source_only"]["target"]["accuracy"]
    for name in ["source_only", "dan", "dann", "cdan"]:
        r = reports[name]
        rows.append({
            "method": name,
            **{f"{d}_val_acc": round(r["source_val_per_domain"][d]["accuracy"], 4) for d in SOURCE_DOMAINS},
            **{f"{d}_val_f1": round(r["source_val_per_domain"][d]["macro_f1"], 4) for d in SOURCE_DOMAINS},
            "mean_source_acc": round(r["source_val_mean"]["accuracy"], 4),
            "mean_source_f1": round(r["source_val_mean"]["macro_f1"], 4),
            "target_acc": round(r["target"]["accuracy"], 4),
            "target_f1": round(r["target"]["macro_f1"], 4),
            "target_acc_change_vs_source_only": round(r["target"]["accuracy"] - source_only_target_acc, 4),
            "domain_separability": round(r["domain_separability"]["domain_separability_accuracy"], 4),
        })
    return rows


def run_class_analysis(reports: dict) -> dict:
    source_only_acc = reports["source_only"]["_per_class_acc_array"]
    out = {}
    for name in ["dan", "dann", "cdan"]:
        method_acc = reports[name]["_per_class_acc_array"]
        deltas = class_deltas_vs_source_only(source_only_acc, method_acc, CLASSES)
        classes_of_interest = list({c for c, _ in deltas["most_improved"] + deltas["most_degraded"]})
        confusions = confusion_for_classes(
            reports[name]["_target_labels"], reports[name]["_target_preds"],
            CLASSES, classes_of_interest)
        out[name] = {**deltas, "confusions": confusions}
    return out


def strip_private_fields(reports: dict) -> dict:
    clean = {}
    for name, r in reports.items():
        clean[name] = {k: v for k, v in r.items() if not k.startswith("_")}
    return clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--controlled_study", choices=["dan", "dann", "none"], default="none",
                         help="Run the controlled alignment-strength study for this method.")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_or_load_splits(base_cfg["pacs"]["root"], base_cfg["pacs"]["train_val_split"], base_cfg["seed"])
    _, val_loaders, _, target_eval_loader = make_loaders(
        splits, batch_size_per_domain=base_cfg["batch"]["per_source_domain"],
        target_batch_size=base_cfg["batch"]["target"])

    ckpt_dir = base_cfg["paths"]["checkpoints_dir"]
    results_dir = base_cfg["paths"]["results_dir"]

    reports = {}
    for name in ["source_only", "dan", "dann", "cdan"]:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Missing checkpoint for '{name}': {ckpt_path}. "
                f"Run `python train.py --config configs/{name}.yaml` first.")
        print(f"Evaluating {name}...")
        reports[name] = evaluate_method(name, ckpt_path, val_loaders, target_eval_loader, device)

    comparison_table = build_comparison_table(reports)
    class_analysis = run_class_analysis(reports)

    final_report = {
        "comparison_table": comparison_table,
        "class_analysis": class_analysis,
        "full_reports": strip_private_fields(reports),
    }

    out_path = os.path.join(results_dir, "final_report.json")
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nSaved final report to {out_path}")
    print("\nComparison table:")
    for row in comparison_table:
        print(row)

    # ---- Controlled design study (Step 6) ----
    if args.controlled_study != "none":
        print(f"\n=== Controlled design study: {args.controlled_study} ===")
        sweep_values = {"dan": [0.1, 1, 10], "dann": [0.25, 0.5, 1]}[args.controlled_study]
        override_key = {"dan": "lambda_mmd", "dann": "max_alpha"}[args.controlled_study]
        method_config_path = f"configs/{args.controlled_study}.yaml"

        study_rows = []
        for val in sweep_values:
            print(f"\n--- {override_key} = {val} ---")
            cfg = load_config(method_config_path, overrides={override_key: val})
            ckpt_path, _ = run_training(cfg, run_name_suffix=f"_{override_key}{val}")
            report = evaluate_method(f"{args.controlled_study}_{override_key}{val}",
                                      ckpt_path, val_loaders, target_eval_loader, device)
            study_rows.append({
                override_key: val,
                "mean_source_acc": round(report["source_val_mean"]["accuracy"], 4),
                "mean_source_f1": round(report["source_val_mean"]["macro_f1"], 4),
                "target_acc": round(report["target"]["accuracy"], 4),
                "target_f1": round(report["target"]["macro_f1"], 4),
                "domain_separability": round(report["domain_separability"]["domain_separability_accuracy"], 4),
            })

        study_path = os.path.join(results_dir, f"controlled_study_{args.controlled_study}.json")
        with open(study_path, "w") as f:
            json.dump(study_rows, f, indent=2)
        print(f"\nControlled study results saved to {study_path}")
        for row in study_rows:
            print(row)


if __name__ == "__main__":
    main()
