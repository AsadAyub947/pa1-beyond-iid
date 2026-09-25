"""
Per-class target accuracy, changes relative to Source-only, and confusion
inspection for the classes with the largest improvement/degradation
(assignment Step 5 / Required Evidence: "per-class target accuracy changes
and selected confusions or failure cases").
"""
import numpy as np
from sklearn.metrics import confusion_matrix


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc[c] = (y_pred[mask] == c).mean()
    return acc


def class_deltas_vs_source_only(source_only_acc: np.ndarray, method_acc: np.ndarray,
                                 class_names: list, top_k: int = 3) -> dict:
    """Returns the classes with the largest improvement and largest
    degradation relative to Source-only, plus the raw per-class deltas."""
    deltas = method_acc - source_only_acc
    order = np.argsort(deltas)
    most_degraded = [(class_names[i], float(deltas[i])) for i in order[:top_k]]
    most_improved = [(class_names[i], float(deltas[i])) for i in order[::-1][:top_k]]
    return {
        "per_class_delta": {class_names[i]: float(deltas[i]) for i in range(len(class_names))},
        "most_improved": most_improved,
        "most_degraded": most_degraded,
    }


def confusion_for_classes(y_true: np.ndarray, y_pred: np.ndarray, class_names: list,
                           classes_of_interest: list) -> dict:
    """Full confusion matrix, plus a convenience readout of what each
    class-of-interest gets most confused with (excluding the diagonal)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = {}
    for cls_name in classes_of_interest:
        c = class_names.index(cls_name)
        row = cm[c].copy()
        true_count = row.sum()
        row_no_diag = row.copy()
        row_no_diag[c] = -1  # exclude self so argmax finds the top confusion
        top_confusion_idx = int(row_no_diag.argmax())
        report[cls_name] = {
            "n_true_examples": int(true_count),
            "n_correct": int(row[c]),
            "top_confused_with": class_names[top_confusion_idx] if row_no_diag[top_confusion_idx] > 0 else None,
            "n_confused_with_top": int(row_no_diag[top_confusion_idx]) if row_no_diag[top_confusion_idx] > 0 else 0,
        }
    return {"confusion_matrix": cm.tolist(), "per_class_report": report}
