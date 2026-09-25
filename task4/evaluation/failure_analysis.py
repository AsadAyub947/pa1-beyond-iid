"""Failure analysis: unknowns that the frozen model *accepts* as known.

Run automatically by ``evaluate_osr.py`` (Vanilla + MLS threshold), or alone:

    python -m evaluation.failure_analysis --model vanilla --score mls --per-group 8

Analysis only – nothing here may feed back into models, score definitions,
hyper-parameters or thresholds.

``SEMANTIC_NEIGHBOURS`` is fixed a priori (before looking at any result): for
each unknown class, the CIFAR-10 labels that a human would consider a
semantically plausible confusion. It is used only to pre-fill the
"plausible vs surprising" column; always confirm it by looking at the images.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from data.cifar10 import CIFAR10_CLASSES

SEMANTIC_NEIGHBOURS = {
    # near unknowns – obvious semantic neighbours in the CIFAR-10 label space
    "bus": {"automobile", "truck"},
    "pickup_truck": {"truck", "automobile"},
    "motorcycle": {"automobile", "truck"},
    "tractor": {"truck", "automobile"},
    "wolf": {"dog", "cat", "deer"},
    "fox": {"dog", "cat", "deer"},
    "leopard": {"cat", "dog", "deer"},
    "camel": {"horse", "deer"},
    # far unknowns – no CIFAR-10 class is a semantic neighbour by construction
    "bottle": set(), "bowl": set(), "chair": set(), "clock": set(),
    "keyboard": set(), "mushroom": set(), "sunflower": set(), "wardrobe": set(),
}


def plausibility(unknown_class: str, predicted: str) -> str:
    return "plausible" if predicted in SEMANTIC_NEIGHBOURS.get(unknown_class, set()) else "surprising"


def accepted_unknowns(unk: dict, u: np.ndarray, tau: float, pred: np.ndarray,
                      msp_conf: np.ndarray | None = None) -> pd.DataFrame:
    """All unknowns with u <= tau, most confidently accepted first."""
    acc = u <= tau
    df = pd.DataFrame({
        "group": unk["group"][acc], "unknown_class": unk["fine_name"][acc],
        "cifar100_test_index": unk["cifar100_index"][acc], "row": np.flatnonzero(acc),
        "predicted_class": [CIFAR10_CLASSES[k] for k in pred[acc]],
        "score_u": u[acc], "threshold_tau": tau, "margin_tau_minus_u": tau - u[acc],
    })
    if msp_conf is not None:
        df["max_softmax_prob"] = msp_conf[acc]
    df["a_priori_semantic_neighbour"] = [plausibility(c, p) for c, p in zip(df.unknown_class, df.predicted_class)]
    df["your_judgement"] = ""  # fill in after inspecting the images
    return df.sort_values("score_u", kind="stable").reset_index(drop=True)


def select_failures(acc_df: pd.DataFrame, group: str, n: int) -> pd.DataFrame:
    """Diverse, informative failures: the most confident accepted image of each
    unknown class first (round-robin over classes), then the next most confident."""
    g = acc_df[acc_df.group == group].copy()
    if g.empty:
        return g
    g["rank_in_class"] = g.groupby("unknown_class").cumcount()
    return g.sort_values(["rank_in_class", "score_u"], kind="stable").head(n).drop(columns="rank_in_class")


def per_class_acceptance(unk: dict, u: np.ndarray, tau: float, pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for group in ("near", "far"):
        for cls in sorted(set(unk["fine_name"][unk["group"] == group])):
            m = unk["fine_name"] == cls
            acc = m & (u <= tau)
            counts = np.bincount(pred[acc], minlength=len(CIFAR10_CLASSES))
            top = int(counts.argmax()) if acc.any() else -1
            rows.append({"group": group, "unknown_class": cls, "n": int(m.sum()),
                         "accept_rate": float(acc.sum() / max(m.sum(), 1)),
                         "top_absorbing_class": CIFAR10_CLASSES[top] if top >= 0 else "",
                         "top_absorbing_share_of_accepted": float(counts[top] / acc.sum()) if acc.any() else 0.0,
                         "argmax_class_all_images": CIFAR10_CLASSES[int(np.bincount(pred[m], minlength=10).argmax())]})
    return pd.DataFrame(rows).sort_values(["group", "accept_rate"], ascending=[False, False]).reset_index(drop=True)


def absorption_matrix(unk: dict, u: np.ndarray, tau: float, pred: np.ndarray) -> pd.DataFrame:
    """Counts of *accepted* unknowns: rows = unknown class, cols = predicted CIFAR-10 class."""
    acc = u <= tau
    classes = [c for g in ("near", "far") for c in sorted(set(unk["fine_name"][unk["group"] == g]))]
    mat = pd.DataFrame(0, index=classes, columns=CIFAR10_CLASSES)
    for c, p in zip(unk["fine_name"][acc], pred[acc]):
        mat.loc[c, CIFAR10_CLASSES[p]] += 1
    mat.index.name = "unknown_class"
    return mat


def plot_failure_grid(images: np.ndarray, near: pd.DataFrame, far: pd.DataFrame, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted = "#0b0b0b", "#52514e"
    ncol = max(len(near), len(far), 1)
    fig, axes = plt.subplots(2, ncol, figsize=(1.55 * ncol + 0.9, 4.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for r, (label, df) in enumerate((("Near unknowns", near), ("Far unknowns", far))):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= len(df):
                if c == 0:  # keep the row label visible and say why the row is empty
                    ax.set_facecolor("#fcfcfb")
                    ax.text(0.5, 0.5, "none\naccepted\nat τ", fontsize=8, color=muted,
                            transform=ax.transAxes, ha="center", va="center")
                else:
                    ax.axis("off")
                continue
            row = df.iloc[c]
            ax.imshow(images[int(row["row"])], interpolation="nearest")
            tag = "plausible" if row["a_priori_semantic_neighbour"] == "plausible" else "surprising"
            ax.set_title(f"{row['unknown_class']}\n→ {row['predicted_class']}", fontsize=8, color=ink, pad=3)
            ax.set_xlabel(f"u={row['score_u']:.2f}\n{tag}", fontsize=7, color=muted, labelpad=2)
        axes[r, 0].set_ylabel(label, fontsize=9, color=ink)
        axes[r, 0].yaxis.label.set_visible(True)
    tau = float((near if len(near) else far)["threshold_tau"].iloc[0]) if len(near) or len(far) else float("nan")
    fig.suptitle(f"{title}  (accept iff u ≤ τ = {tau:.3f})", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    import evaluate_osr as ev  # reuse the loaders / score definitions

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vanilla")
    ap.add_argument("--score", default="mls")
    ap.add_argument("--per-group", type=int, default=8)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--override", nargs="*", default=[], help="config overrides, e.g. data.root=...")
    args = ap.parse_args()
    ev.run_failure_analysis(args.model, args.score, args.per_group, args.results_dir, args.override)


if __name__ == "__main__":
    main()
