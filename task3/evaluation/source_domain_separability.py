"""
Source-domain separability (assignment Step 4): balanced features from the
THREE SOURCE validation domains only (never Sketch), 70/30 split (seed 6304),
multinomial logistic regression (C=1) predicting which of the 3 source
domains a feature came from. Held-out accuracy is the score; chance = 33.3%.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 6304


def source_domain_separability_score(features_by_domain: dict, seed: int = SEED) -> dict:
    """features_by_domain: {domain_name: (N_d, D) array}, one entry per
    source domain (exactly 3 expected: photo, art_painting, cartoon)."""
    domains = sorted(features_by_domain.keys())
    assert len(domains) == 3, f"Expected exactly 3 source domains, got {domains}"

    n = min(len(features_by_domain[d]) for d in domains)  # balance classes
    rng = np.random.RandomState(seed)

    X_parts, y_parts = [], []
    for label, d in enumerate(domains):
        idx = rng.choice(len(features_by_domain[d]), size=n, replace=False)
        X_parts.append(features_by_domain[d][idx])
        y_parts.append(np.full(n, label))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)  # multinomial by default for >2 classes
    clf.fit(X_train, y_train)
    held_out_acc = float(clf.score(X_test, y_test))

    return {
        "source_domain_separability_accuracy": held_out_acc,
        "n_per_domain_used": int(n),
        "chance_level": 1.0 / 3.0,
        "domain_label_order": domains,
    }
