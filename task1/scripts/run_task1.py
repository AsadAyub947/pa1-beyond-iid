"""
End-to-end Task 1 pipeline orchestrator (Memory-Optimized for Colab).
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.datasets import STL10

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add task1/ to path

from analysis.evaluate_bias import (
    compute_metrics,
    patch_shuffle_report,
    prediction_consistency,
    run_color_bias_report,
    shape_texture_bias,
)
from analysis.feature_similarity import cosine_stability
from analysis.representation import fit_projection, plot_projection
from data.transforms import (
    grayscale_transform,
    hue_rotate_transform,
    patch_shuffle_transform,
    to_common_image,
    translate_transform,
)
from models.backbones import build_backbone, train_linear_head


def set_all_seeds(seed):
  import random

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def load_images_as_tensor(dataset, indices):
  """Return a (N,3,224,224) tensor of common-representation images for the given dataset indices."""
  imgs = []
  for i in indices:
    pil_img = Image.fromarray(dataset.data[i].transpose(1, 2, 0))
    imgs.append(to_common_image(pil_img))
  return torch.stack(imgs)


@torch.no_grad()
def batched_extract(backbone, img01_all, batch_size=32):
  """Memory-safe feature extraction loop with explicit autograd suppression."""
  feats = []
  for start in range(0, img01_all.shape[0], batch_size):
    batch = img01_all[start : start + batch_size]
    f = backbone.extract_features(batch)
    feats.append(f.detach().cpu())
    del batch, f
  return torch.cat(feats, dim=0)


def predict_with_head(head, feats, device):
  with torch.no_grad():
    logits = head(feats.to(device))
    probs = logits.softmax(dim=-1).cpu().numpy()
    preds = probs.argmax(axis=1)
  return preds, probs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/config.yaml")
  args = parser.parse_args()

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  seed = cfg["seed"]
  set_all_seeds(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  results_dir = cfg["paths"]["results_dir"]
  os.makedirs(results_dir, exist_ok=True)

  subset_path = os.path.join(results_dir, "subset_indices.json")
  if not os.path.exists(subset_path):
    raise FileNotFoundError(
        f"{subset_path} not found. Run `python data/make_subset.py` first."
    )
  with open(subset_path) as f:
    subset = json.load(f)

  cue_meta_path = os.path.join(results_dir, "cue_conflicts", "metadata.json")
  if not os.path.exists(cue_meta_path):
    raise FileNotFoundError(
        f"{cue_meta_path} not found. Run `python"
        " data/make_cue_conflicts.py` first."
    )
  with open(cue_meta_path) as f:
    cue_meta = json.load(f)

  class_names = cfg["dataset"]["class_names"]
  num_classes = cfg["dataset"]["num_classes"]

  print("Loading STL-10 datasets...")
  train_ds = STL10(root=cfg["dataset"]["root"], split="train", download=True)
  test_ds = STL10(root=cfg["dataset"]["root"], split="test", download=True)

  train_idx, val_idx = subset["train_idx"], subset["val_idx"]
  test_idx = subset["test_subset_idx"]
  y_train = np.array(train_ds.labels)[train_idx]
  y_val = np.array(train_ds.labels)[val_idx]
  y_test = np.array(test_ds.labels)[test_idx]

  print("Materializing test & transformed image tensors...")
  clean_imgs = load_images_as_tensor(test_ds, test_idx)
  gray_imgs = torch.stack([grayscale_transform(im) for im in clean_imgs])
  hue_shift = cfg["color_bias"]["hue_shift"]
  hue_imgs = torch.stack(
      [hue_rotate_transform(im, hue_shift) for im in clean_imgs]
  )
  grid = cfg["patch_shuffle"]["grid"]
  shuffled_imgs = torch.stack([
      patch_shuffle_transform(im, grid=grid, seed=seed, index=i)
      for i, im in enumerate(clean_imgs)
  ])

  displacements = cfg["translation"]["displacements"]
  directions = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
  translated_imgs = {}
  for disp in displacements:
    if disp == 0:
      translated_imgs[disp] = {d: clean_imgs for d in directions}
      continue
    translated_imgs[disp] = {}
    for dname, (sx, sy) in directions.items():
      dx, dy = sx * disp, sy * disp
      translated_imgs[disp][dname] = torch.stack(
          [translate_transform(im, dx, dy) for im in clean_imgs]
      )

  cue_conflict_dir = os.path.join(results_dir, "cue_conflicts")
  cue_records = cue_meta["conflicts"]
  cue_imgs = torch.stack([
      torch.load(os.path.join(cue_conflict_dir, rec["file"]))
      for rec in cue_records
  ])
  cue_content_labels = np.array([r["content_label"] for r in cue_records])
  cue_style_labels = np.array([r["style_label"] for r in cue_records])

  all_backbone_results = {}
  linear_cfg = cfg["linear_head"]
  clean_predictions_by_backbone = {}

  for backbone_name in ["resnet50", "vit_b_16", "clip_vit_b32"]:
    print(f"\n=== Backbone: {backbone_name} ===")

    # Clear memory prior to building model
    gc.collect()
    torch.cuda.empty_cache()

    backbone = build_backbone(backbone_name, device)
    feat_dim = backbone.feat_dim

    print("Extracting train/val features...")
    train_imgs = load_images_as_tensor(train_ds, train_idx)
    val_imgs = load_images_as_tensor(train_ds, val_idx)
    train_feats = batched_extract(
        backbone, train_imgs, batch_size=linear_cfg["batch_size"]
    )
    val_feats = batched_extract(
        backbone, val_imgs, batch_size=linear_cfg["batch_size"]
    )

    # Free raw image tensors from memory immediately
    del train_imgs, val_imgs
    gc.collect()

    print("Training linear head...")
    head, best_val_acc = train_linear_head(
        train_feats,
        torch.tensor(y_train, dtype=torch.long),
        val_feats,
        torch.tensor(y_val, dtype=torch.long),
        feat_dim,
        num_classes,
        device,
        seed=seed,
        lr=linear_cfg["lr"],
        weight_decay=linear_cfg["weight_decay"],
        max_epochs=linear_cfg["max_epochs"],
        patience=linear_cfg["early_stop_patience"],
        batch_size=linear_cfg["batch_size"],
    )
    print(f"Best val acc: {best_val_acc:.4f}")

    def eval_condition(imgs):
      feats = batched_extract(
          backbone, imgs, batch_size=linear_cfg["batch_size"]
      )
      preds, probs = predict_with_head(head, feats, device)
      return feats, preds, probs

    feat_clean, pred_clean, probs_clean = eval_condition(clean_imgs)
    clean_predictions_by_backbone[backbone_name] = pred_clean
    feat_gray, pred_gray, probs_gray = eval_condition(gray_imgs)
    feat_hue, pred_hue, probs_hue = eval_condition(hue_imgs)
    feat_shuf, pred_shuf, probs_shuf = eval_condition(shuffled_imgs)
    feat_cue, pred_cue, probs_cue = eval_condition(cue_imgs)

    preds_by_disp, probs_by_disp, feats_by_disp = {}, {}, {}
    for disp in displacements:
      dir_preds, dir_probs, dir_feats = [], [], []
      for dname in directions:
        f, p, pr = eval_condition(translated_imgs[disp][dname])
        dir_preds.append(p)
        dir_probs.append(pr)
        dir_feats.append(f)
      preds_by_disp[disp] = dir_preds
      probs_by_disp[disp] = dir_probs
      feats_by_disp[disp] = torch.stack(dir_feats).mean(dim=0)

    clean_metrics = compute_metrics(y_test, pred_clean, probs_clean)
    color_report = run_color_bias_report(
        y_test,
        pred_clean,
        probs_clean,
        pred_gray,
        probs_gray,
        pred_hue,
        probs_hue,
        extra_name="hue_rotation",
    )
    shape_report = shape_texture_bias(
        pred_cue, cue_content_labels, cue_style_labels
    )

    translation_rows = []
    for disp in displacements:
      per_dir_metrics = [
          compute_metrics(
              y_test, preds_by_disp[disp][k], probs_by_disp[disp][k]
          )
          for k in range(len(directions))
      ]
      per_dir_consistency = [
          prediction_consistency(pred_clean, preds_by_disp[disp][k])
          for k in range(len(directions))
      ]
      translation_rows.append({
          "displacement_px": disp,
          "top1_acc": float(np.mean([m["top1_acc"] for m in per_dir_metrics])),
          "macro_f1": float(np.mean([m["macro_f1"] for m in per_dir_metrics])),
          "mean_max_conf": float(
              np.mean([m["mean_max_conf"] for m in per_dir_metrics])
          ),
          "consistency_vs_clean": float(np.mean(per_dir_consistency)),
      })

    patch_report = patch_shuffle_report(
        y_test, pred_clean, probs_clean, pred_shuf, probs_shuf
    )

    cue_content_imgs = torch.stack([
        to_common_image(
            Image.fromarray(
                test_ds.data[r["source_content_idx"]].transpose(1, 2, 0)
            )
        )
        for r in cue_records
    ])
    feat_cue_content = batched_extract(
        backbone, cue_content_imgs, batch_size=linear_cfg["batch_size"]
    )

    rep_stability = {
        "grayscale": cosine_stability(feat_clean, feat_gray),
        "hue_rotation": cosine_stability(feat_clean, feat_hue),
        "cue_conflict_vs_source_content": cosine_stability(
            feat_cue_content, feat_cue
        ),
        "patch_shuffle": cosine_stability(feat_clean, feat_shuf),
    }
    for disp in displacements:
      if disp == 0:
        continue
      rep_stability[f"translation_{disp}px"] = cosine_stability(
          feat_clean, feats_by_disp[disp]
      )

    rep_cfg = cfg["representation"]
    plots_dir = os.path.join(results_dir, "plots", backbone_name)
    os.makedirs(plots_dir, exist_ok=True)
    conditions_to_plot = {
        "grayscale": (feat_clean, feat_gray, y_test, y_test),
        "hue_rotation": (feat_clean, feat_hue, y_test, y_test),
        "patch_shuffle": (feat_clean, feat_shuf, y_test, y_test),
        "cue_conflict": (
            feat_cue_content,
            feat_cue,
            cue_content_labels,
            cue_content_labels,
        ),
        f"translation_{displacements[-1]}px": (
            feat_clean,
            feats_by_disp[displacements[-1]],
            y_test,
            y_test,
        ),
    }
    for cond_name, (fc, ft, yc, yt) in conditions_to_plot.items():
      combined = torch.cat([fc, ft], dim=0).numpy()
      combined_classes = np.concatenate([yc, yt])
      combined_conditions = np.array(
          ["clean"] * len(yc) + [cond_name] * len(yt)
      )
      coords = fit_projection(
          combined,
          method=rep_cfg["method"],
          seed=seed,
          tsne_perplexity=rep_cfg["tsne_perplexity"],
          umap_n_neighbors=rep_cfg["umap_n_neighbors"],
          umap_min_dist=rep_cfg["umap_min_dist"],
      )
      plot_projection(
          coords,
          combined_classes,
          combined_conditions,
          class_names,
          title=f"{backbone_name}: clean vs {cond_name}",
          save_path=os.path.join(plots_dir, f"{cond_name}.png"),
      )

    all_backbone_results[backbone_name] = {
        "val_acc_at_selection": best_val_acc,
        "clean_baseline": clean_metrics,
        "color_bias": color_report,
        "shape_texture": shape_report,
        "translation_curve": translation_rows,
        "patch_shuffle": patch_report,
        "representation_stability": rep_stability,
    }

    # Delete backbone from CUDA memory before loading the next model
    del backbone, head
    gc.collect()
    torch.cuda.empty_cache()

  print("\n=== CLIP zero-shot ===")
  clip_backbone = build_backbone("clip_vit_b32", device)
  prompt_template = cfg["clip_zero_shot"]["prompt_template"]

  def zero_shot_eval(imgs):
    preds_all, confs_all, probs_all = [], [], []
    bs = linear_cfg["batch_size"]
    for start in range(0, imgs.shape[0], bs):
      batch = imgs[start : start + bs]
      pred, conf, probs = clip_backbone.zero_shot_predict(
          batch, class_names, prompt_template
      )
      preds_all.append(pred.numpy())
      confs_all.append(conf.numpy())
      probs_all.append(probs.numpy())
    return np.concatenate(preds_all), np.concatenate(probs_all)

  zs_pred_clean, zs_probs_clean = zero_shot_eval(clean_imgs)
  zs_metrics_clean = compute_metrics(y_test, zs_pred_clean, zs_probs_clean)
  zs_pred_gray, zs_probs_gray = zero_shot_eval(gray_imgs)
  zs_pred_hue, zs_probs_hue = zero_shot_eval(hue_imgs)
  zs_color_report = run_color_bias_report(
      y_test,
      zs_pred_clean,
      zs_probs_clean,
      zs_pred_gray,
      zs_probs_gray,
      zs_pred_hue,
      zs_probs_hue,
      extra_name="hue_rotation",
  )
  zs_pred_cue, zs_probs_cue = zero_shot_eval(cue_imgs)
  zs_shape_report = shape_texture_bias(
      zs_pred_cue, cue_content_labels, cue_style_labels
  )
  zs_pred_shuf, zs_probs_shuf = zero_shot_eval(shuffled_imgs)
  zs_patch_report = patch_shuffle_report(
      y_test, zs_pred_clean, zs_probs_clean, zs_pred_shuf, zs_probs_shuf
  )

  clip_head_pred_clean = clean_predictions_by_backbone["clip_vit_b32"]
  zero_shot_vs_head_agreement = prediction_consistency(
      clip_head_pred_clean, zs_pred_clean
  )

  all_results = {
      "backbones": all_backbone_results,
      "clip_zero_shot": {
          "clean_baseline": zs_metrics_clean,
          "color_bias": zs_color_report,
          "shape_texture": zs_shape_report,
          "patch_shuffle": zs_patch_report,
          "agreement_with_trained_clip_head_on_clean": (
              zero_shot_vs_head_agreement
          ),
      },
  }

  out_path = os.path.join(results_dir, "task1_results.json")
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
  print(f"\nAll results saved to {out_path}")
  print(f"Plots saved under {os.path.join(results_dir, 'plots')}")


if __name__ == "__main__":
  main()