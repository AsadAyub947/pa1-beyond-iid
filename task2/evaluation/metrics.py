import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def domain_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def mean_across_domains(per_domain_metrics: dict) -> dict:
    """per_domain_metrics: {domain_name: {"accuracy":..., "macro_f1":...}}"""
    accs = [m["accuracy"] for m in per_domain_metrics.values()]
    f1s = [m["macro_f1"] for m in per_domain_metrics.values()]
    return {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
