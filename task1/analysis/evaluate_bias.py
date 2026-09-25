"""
Evaluation routines for each experimental step (clean baseline, color bias,
shape-vs-texture cue conflict, translation, patch shuffle).

Every function here operates on already-extracted PREDICTIONS (and, where
relevant, PROBABILITIES) rather than re-implementing dataset iteration, so the
same functions apply uniformly to the three trained linear heads and to
zero-shot CLIP.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> dict:
    """Top-1 accuracy, macro-F1, and mean maximum softmax confidence."""
    top1 = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    mean_max_conf = float(probs.max(axis=1).mean())
    return {"top1_acc": top1, "macro_f1": macro_f1, "mean_max_conf": mean_max_conf}


def prediction_consistency(pred_clean: np.ndarray, pred_transformed: np.ndarray) -> float:
    """Fraction of images whose predicted class is unchanged after an
    intervention (Consistency(delta) in the assignment, generalized to any
    transform)."""
    return float((pred_clean == pred_transformed).mean())


def accuracy_delta(metrics_transformed: dict, metrics_clean: dict) -> float:
    return metrics_transformed["top1_acc"] - metrics_clean["top1_acc"]


def run_color_bias_report(y_true, pred_clean, probs_clean,
                           pred_gray, probs_gray,
                           pred_extra, probs_extra, extra_name: str) -> dict:
    clean_m = compute_metrics(y_true, pred_clean, probs_clean)
    gray_m = compute_metrics(y_true, pred_gray, probs_gray)
    extra_m = compute_metrics(y_true, pred_extra, probs_extra)
    return {
        "clean": clean_m,
        "grayscale": {
            **gray_m,
            "accuracy_delta_vs_clean": accuracy_delta(gray_m, clean_m),
            "consistency_vs_clean": prediction_consistency(pred_clean, pred_gray),
        },
        extra_name: {
            **extra_m,
            "accuracy_delta_vs_clean": accuracy_delta(extra_m, clean_m),
            "consistency_vs_clean": prediction_consistency(pred_clean, pred_extra),
        },
    }


def shape_texture_bias(preds: np.ndarray, content_labels: np.ndarray,
                        style_labels: np.ndarray) -> dict:
    """Classify each cue-conflict prediction as shape (content label), texture
    (style label), or other; report Shape Bias(%) and Coverage(%)."""
    n_total = len(preds)
    is_shape = preds == content_labels
    is_texture = (preds == style_labels) & (~is_shape)
    n_shape = int(is_shape.sum())
    n_texture = int(is_texture.sum())
    n_other = n_total - n_shape - n_texture

    denom = n_shape + n_texture
    shape_bias_pct = 100.0 * n_shape / denom if denom > 0 else float("nan")
    coverage_pct = 100.0 * denom / n_total if n_total > 0 else float("nan")

    return {
        "n_total": n_total,
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "shape_bias_pct": shape_bias_pct,
        "coverage_pct": coverage_pct,
    }


def translation_curve(y_true, preds_by_displacement: dict, probs_by_displacement: dict,
                       pred_clean: np.ndarray) -> list:
    """preds_by_displacement: {displacement_px: pred_array_averaged_over_4_directions}
    (caller is responsible for averaging predictions/metrics across the four
    cardinal directions before calling this, per the assignment's instruction
    to 'average the results across directions')."""
    rows = []
    for disp in sorted(preds_by_displacement.keys()):
        preds = preds_by_displacement[disp]
        probs = probs_by_displacement[disp]
        m = compute_metrics(y_true, preds, probs)
        rows.append({
            "displacement_px": disp,
            "top1_acc": m["top1_acc"],
            "macro_f1": m["macro_f1"],
            "mean_max_conf": m["mean_max_conf"],
            "consistency_vs_clean": prediction_consistency(pred_clean, preds),
        })
    return rows


def patch_shuffle_report(y_true, pred_clean, probs_clean, pred_shuffled, probs_shuffled) -> dict:
    clean_m = compute_metrics(y_true, pred_clean, probs_clean)
    shuf_m = compute_metrics(y_true, pred_shuffled, probs_shuffled)
    return {
        "clean": clean_m,
        "shuffled": {
            **shuf_m,
            "accuracy_drop": clean_m["top1_acc"] - shuf_m["top1_acc"],
            "consistency_vs_clean": prediction_consistency(pred_clean, pred_shuffled),
        },
    }
