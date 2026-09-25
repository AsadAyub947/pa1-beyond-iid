"""Open-set metrics. Convention: unknown = positive class, u(x) larger = more novel."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """Probability that a random unknown has a larger u than a random known."""
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    return float(roc_auc_score(y, np.r_[u_known, u_unknown]))


def roc(u_known: np.ndarray, u_unknown: np.ndarray):
    """ROC with unknown as positive: returns (FPR = known rejected, TPR = unknown rejected)."""
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    fpr, tpr, _ = roc_curve(y, np.r_[u_known, u_unknown])
    return fpr, tpr


def acceptance_rate(u: np.ndarray, tau: float) -> float:
    """Fraction accepted as known under the rule accept x iff u(x) <= tau."""
    return float(np.mean(np.asarray(u) <= tau)) if len(u) else float("nan")


def rejection_rate(u: np.ndarray, tau: float) -> float:
    return 1.0 - acceptance_rate(u, tau) if len(u) else float("nan")


def closed_set_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.asarray(logits).argmax(1) == np.asarray(labels)))


def osr_report(u_val: np.ndarray, u_test: np.ndarray, u_near: np.ndarray, u_far: np.ndarray,
               tau: float) -> dict:
    """All required numbers for one (model, score) pair.

    ``fpr95_*`` follows the task convention: the fraction of unknown examples
    incorrectly *accepted* at the validation-calibrated 95 %-TPR threshold
    (known = positive for TPR), i.e. ``1 - rejection``.
    """
    u_all = np.r_[u_near, u_far]
    return {
        "auroc_near": auroc(u_test, u_near),
        "auroc_far": auroc(u_test, u_far),
        "auroc_all": auroc(u_test, u_all),
        "tau": float(tau),
        "val_accept": acceptance_rate(u_val, tau),
        "test_accept": acceptance_rate(u_test, tau),
        "near_reject": rejection_rate(u_near, tau),
        "far_reject": rejection_rate(u_far, tau),
        "all_reject": rejection_rate(u_all, tau),
        "fpr95_near": acceptance_rate(u_near, tau),
        "fpr95_far": acceptance_rate(u_far, tau),
        "fpr95_all": acceptance_rate(u_all, tau),
    }
