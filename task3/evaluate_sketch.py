"""
Final evaluation stage. This is the ONLY script in Task 3 that loads Sketch
images or labels -- run it ONLY after ERM (= Task 2's Source-only checkpoint),
DAN-DG, and SAM checkpoints all already exist and are considered final.

Usage:
    python evaluate_sketch.py --base_config configs/base.yaml
    python evaluate_sketch.py --base_config configs/base.yaml --controlled_study dan_dg
    python evaluate_sketch.py --base_config configs/base.yaml --controlled_study sam
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared.pacs_protocol import build_or_load_splits, make_loaders, SOURCE_DOMAINS
from shared.pacs import CLASSES, PACSDomainDataset, eval_transform
from models.backbone import ResNet18Backbone, FEAT_DIM
from models.classifier_head import ClassifierHead, NUM_CLASSES
from methods.erm import load_erm_checkpoint
from evaluation.domain_metrics import (compute_domain_metrics, mean_across_domains, worst_across_domains,
                                        per_class_accuracy, class_deltas_vs_erm, confusion_for_classes)
from evaluation.source_domain_separability import source_domain_separability_score
from evaluation.sharpness import build_fixed_sharpness_batch, compute_sharpness
from train import load_config, run_training


@torch.no_grad()
def get_preds_labels_features(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    preds, labels, feats = [], [], []
    for x, y in loader:
        x = x.to(device)
        feat = backbone(x)
        logits = head(feat)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(y.numpy())
        feats.append(feat.cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(feats, axis=0)


def load_method_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(FEAT_DIM, NUM_CLASSES).to(device)
    backbone.load_state_dict(ckpt["state"]["backbone"])
    head.load_state_dict(ckpt["state"]["head"])
    return backbone, head


def evaluate_model(name: str, backbone, head, val_loaders: dict, target_eval_loader,
                    fixed_sharp_x, fixed_sharp_y, device) -> dict:
    per_domain = {}
    feats_by_domain = {}
    for domain, loader in val_loaders.items():
        preds, labels, feats = get_preds_labels_features(backbone, head, loader, device)
        per_domain[domain] = compute_domain_metrics(labels, preds)
        feats_by_domain[domain] = feats

    mean_source = mean_across_domains(per_domain)
    worst_source = worst_across_domains(per_domain)

    target_preds, target_labels, _ = get_preds_labels_features(backbone, head, target_eval_loader, device)
    target_report = compute_domain_metrics(target_labels, target_preds)

    sep = source_domain_separability_score(feats_by_domain)
    sharp = compute_sharpness(backbone, head, fixed_sharp_x, fixed_sharp_y)

    per_class_acc = per_class_accuracy(target_labels, target_preds, NUM_CLASSES)

    return {
        "run_name": name,
        "source_val_per_domain": per_domain,
        "source_val_mean": mean_source,
        "source_val_worst": worst_source,
        "sketch": target_report,
        "source_domain_separability": sep,
        "sharpness_delta": sharp,
        "per_class_sketch_accuracy": {CLASSES[i]: (None if np.isnan(per_class_acc[i]) else float(per_class_acc[i]))
                                       for i in range(NUM_CLASSES)},
        "_sketch_preds": target_preds, "_sketch_labels": target_labels,
        "_per_class_acc_array": per_class_acc,
    }


def build_comparison_table(reports: dict) -> list:
    rows = []
    erm_sketch_acc = reports["erm"]["sketch"]["accuracy"]
    for name in ["erm", "dan_dg", "sam"]:
        r = reports[name]
        rows.append({
            "method": name,
            **{f"{d}_val_acc": round(r["source_val_per_domain"][d]["accuracy"], 4) for d in SOURCE_DOMAINS},
            "mean_source_acc": round(r["source_val_mean"]["accuracy"], 4),
            "mean_source_f1": round(r["source_val_mean"]["macro_f1"], 4),
            "worst_source_acc": round(r["source_val_worst"]["accuracy"], 4),
            "worst_source_domain": r["source_val_worst"]["accuracy_domain"],
            "sketch_acc": round(r["sketch"]["accuracy"], 4),
            "sketch_f1": round(r["sketch"]["macro_f1"], 4),
            "sketch_acc_change_vs_erm": round(r["sketch"]["accuracy"] - erm_sketch_acc, 4),
            "source_domain_separability": round(r["source_domain_separability"]["source_domain_separability_accuracy"], 4),
            "sharpness_delta": round(r["sharpness_delta"], 6),
        })
    return rows


def run_class_analysis(reports: dict) -> dict:
    erm_acc = reports["erm"]["_per_class_acc_array"]
    out = {}
    for name in ["dan_dg", "sam"]:
        method_acc = reports[name]["_per_class_acc_array"]
        deltas = class_deltas_vs_erm(erm_acc, method_acc, CLASSES)
        classes_of_interest = list({c for c, _ in deltas["most_improved"] + deltas["most_degraded"]})
        confusions = confusion_for_classes(reports[name]["_sketch_labels"], reports[name]["_sketch_preds"],
                                            CLASSES, classes_of_interest)
        out[name] = {**deltas, "confusions": confusions}
    return out


def try_load_task2_comparison(results_dir: str) -> dict:
    """Optionally pull in Task 2's final_report.json so DAN-DG (target-free)
    can be directly compared with Task 2's DAN (target-aware) -- Research
    Question 4 asks for exactly this."""
    task2_report_path = os.path.join(results_dir, "..", "..", "task2", "results", "final_report.json")
    if not os.path.exists(task2_report_path):
        return {"note": f"Task 2 final_report.json not found at {task2_report_path}; "
                         f"run Task 2's evaluate_final.py first for a direct comparison."}
    with open(task2_report_path) as f:
        task2_report = json.load(f)
    task2_dan_row = next((r for r in task2_report["comparison_table"] if r["method"] == "dan"), None)
    task2_source_only_row = next((r for r in task2_report["comparison_table"] if r["method"] == "source_only"), None)
    return {
        "task2_source_only_target_acc": task2_source_only_row["target_acc"] if task2_source_only_row else None,
        "task2_dan_target_acc_change_vs_source_only": task2_dan_row["target_acc_change_vs_source_only"] if task2_dan_row else None,
        "task2_dan_domain_separability": task2_dan_row["domain_separability"] if task2_dan_row else None,
        "note": "Compare these (target-AWARE DAN, Task 2) against dan_dg's "
                "sketch_acc_change_vs_erm and source_domain_separability above "
                "(target-FREE DAN-DG, Task 3) for Research Question 4.",
    }


def strip_private_fields(reports: dict) -> dict:
    return {name: {k: v for k, v in r.items() if not k.startswith("_")} for name, r in reports.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--controlled_study", choices=["dan_dg", "sam", "none"], default="none")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = build_or_load_splits(base_cfg["pacs"]["root"], base_cfg["pacs"]["train_val_split"], base_cfg["seed"])
    _, val_loaders, _, target_eval_loader = make_loaders(
        splits, batch_size_per_domain=base_cfg["batch"]["per_source_domain"], target_batch_size=1)

    # Fixed sharpness batch: built once, seeded, from source-val DATASETS directly
    # (not loaders) so indexing is exactly reproducible.
    val_source_datasets = {}
    for domain in SOURCE_DOMAINS:
        d = splits["domains"][domain]
        val_source_datasets[domain] = PACSDomainDataset(d["filepaths"], d["labels"], d["val_idx"], eval_transform())
    fixed_sharp_x, fixed_sharp_y = build_fixed_sharpness_batch(val_source_datasets, device=device)

    results_dir = base_cfg["paths"]["results_dir"]
    ckpt_dir = base_cfg["paths"]["checkpoints_dir"]

    print("Loading ERM (= Task 2 Source-only checkpoint, NOT retrained)...")
    erm_backbone, erm_head = load_erm_checkpoint(base_cfg["paths"]["task2_source_only_checkpoint"], device)

    reports = {"erm": evaluate_model("erm", erm_backbone, erm_head, val_loaders, target_eval_loader,
                                      fixed_sharp_x, fixed_sharp_y, device)}

    for name in ["dan_dg", "sam"]:
        ckpt_path = os.path.join(ckpt_dir, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint for '{name}': {ckpt_path}. "
                                     f"Run `python train.py --config configs/{name}.yaml` first.")
        backbone, head = load_method_checkpoint(ckpt_path, device)
        reports[name] = evaluate_model(name, backbone, head, val_loaders, target_eval_loader,
                                        fixed_sharp_x, fixed_sharp_y, device)
        print(f"Evaluated {name}.")

    comparison_table = build_comparison_table(reports)
    class_analysis = run_class_analysis(reports)
    task2_comparison = try_load_task2_comparison(results_dir)

    final_report = {
        "comparison_table": comparison_table,
        "class_analysis": class_analysis,
        "task2_comparison": task2_comparison,
        "full_reports": strip_private_fields(reports),
    }
    out_path = os.path.join(results_dir, "final_report.json")
    with open(out_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"\nSaved final report to {out_path}")
    print("\nComparison table:")
    for row in comparison_table:
        print(row)
    print("\nTask 2 comparison note:", task2_comparison.get("note"))

    if args.controlled_study != "none":
        print(f"\n=== Controlled design study: {args.controlled_study} ===")
        sweep_values = {"dan_dg": [0.1, 1, 10], "sam": [0.01, 0.05, 0.1]}[args.controlled_study]
        override_key = {"dan_dg": "lambda_dg", "sam": "rho"}[args.controlled_study]
        config_path = f"configs/{args.controlled_study}.yaml"

        study_rows = []
        for val in sweep_values:
            print(f"\n--- {override_key} = {val} ---")
            cfg = load_config(config_path, overrides={override_key: val})
            ckpt_path, _ = run_training(cfg, run_name_suffix=f"_{override_key}{val}")
            backbone, head = load_method_checkpoint(ckpt_path, device)
            report = evaluate_model(f"{args.controlled_study}_{override_key}{val}", backbone, head,
                                     val_loaders, target_eval_loader, fixed_sharp_x, fixed_sharp_y, device)
            study_rows.append({
                override_key: val,
                "mean_source_acc": round(report["source_val_mean"]["accuracy"], 4),
                "mean_source_f1": round(report["source_val_mean"]["macro_f1"], 4),
                "sketch_acc": round(report["sketch"]["accuracy"], 4),
                "sketch_f1": round(report["sketch"]["macro_f1"], 4),
                "source_domain_separability": round(
                    report["source_domain_separability"]["source_domain_separability_accuracy"], 4),
                "sharpness_delta": round(report["sharpness_delta"], 6),
            })

        study_path = os.path.join(results_dir, f"controlled_study_{args.controlled_study}.json")
        with open(study_path, "w") as f:
            json.dump(study_rows, f, indent=2)
        print(f"\nControlled study saved to {study_path}")
        for row in study_rows:
            print(row)
        print("\nReminder: the main comparison above must keep using lambda_dg=1 / rho=0.05 "
              "regardless of what this sweep shows -- do not pick a 'winner' from Sketch results.")


if __name__ == "__main__":
    main()
