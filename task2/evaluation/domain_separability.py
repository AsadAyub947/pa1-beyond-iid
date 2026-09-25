"""
Domain-separability diagnostic (assignment Step 5): freeze a backbone, collect
EQUAL numbers of source-validation and target features, split 70/30 (seed
6304), and train a balanced logistic-regression probe (C=1) to distinguish
source from target. Its held-out accuracy is the domain separability score;
50% == chance == domains are indistinguishable to a linear probe.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def domain_separability_score(source_features: np.ndarray, target_features: np.ndarray,
                               seed: int = SEED) -> dict:
    """source_features, target_features: (N, D) each, with N already equalized
    by the caller (subsample the larger one to match the smaller)."""
    n = min(len(source_features), len(target_features))
    rng = np.random.RandomState(seed)
    src_idx = rng.choice(len(source_features), size=n, replace=False)
    tgt_idx = rng.choice(len(target_features), size=n, replace=False)
    X = np.concatenate([source_features[src_idx], target_features[tgt_idx]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])  # 0 = source, 1 = target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test))

    return {
        "domain_separability_accuracy": held_out_acc,
        "n_per_domain_used": int(n),
        "chance_level": 0.5,
    }
