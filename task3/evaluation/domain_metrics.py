import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_domain_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def mean_across_domains(per_domain_metrics: dict) -> dict:
    accs = [m["accuracy"] for m in per_domain_metrics.values()]
    f1s = [m["macro_f1"] for m in per_domain_metrics.values()]
    return {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}


def worst_across_domains(per_domain_metrics: dict) -> dict:
    accs = {d: m["accuracy"] for d, m in per_domain_metrics.items()}
    f1s = {d: m["macro_f1"] for d, m in per_domain_metrics.items()}
    worst_acc_domain = min(accs, key=accs.get)
    worst_f1_domain = min(f1s, key=f1s.get)
    return {
        "accuracy": accs[worst_acc_domain], "accuracy_domain": worst_acc_domain,
        "macro_f1": f1s[worst_f1_domain], "macro_f1_domain": worst_f1_domain,
    }


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.full(num_classes, np.nan)
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            acc[c] = (y_pred[mask] == c).mean()
    return acc


def class_deltas_vs_erm(erm_acc: np.ndarray, method_acc: np.ndarray,
                         class_names: list, top_k: int = 3) -> dict:
    deltas = method_acc - erm_acc
    order = np.argsort(deltas)
    return {
        "per_class_delta": {class_names[i]: float(deltas[i]) for i in range(len(class_names))},
        "most_improved": [(class_names[i], float(deltas[i])) for i in order[::-1][:top_k]],
        "most_degraded": [(class_names[i], float(deltas[i])) for i in order[:top_k]],
    }


def confusion_for_classes(y_true, y_pred, class_names: list, classes_of_interest: list) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = {}
    for cls_name in classes_of_interest:
        c = class_names.index(cls_name)
        row = cm[c].copy()
        row_no_diag = row.copy()
        row_no_diag[c] = -1
        top_idx = int(row_no_diag.argmax())
        report[cls_name] = {
            "n_true_examples": int(row.sum()),
            "n_correct": int(row[c]),
            "top_confused_with": class_names[top_idx] if row_no_diag[top_idx] > 0 else None,
            "n_confused_with_top": int(row_no_diag[top_idx]) if row_no_diag[top_idx] > 0 else 0,
        }
    return {"confusion_matrix": cm.tolist(), "per_class_report": report}
