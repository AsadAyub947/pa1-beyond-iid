"""Common open-set evaluation for every frozen model and score.

    python evaluate_osr.py                 # vanilla, gcsc, proser (+ rpl if its cache exists)
    python evaluate_osr.py --models vanilla gcsc proser rpl

Reads only ``cache/<model>/{known,unknown}.npz`` (+ the CIFAR-100 test images for
the failure figure). Protocol, in this order:

1. Known-only stage – for every (model, score): compute u(x) on CIFAR-10
   val/test, set tau = 95th percentile of u on *validation*, record CSA. All
   thresholds are written to results/thresholds.json before any unknown is read.
2. Unknown stage – score the fixed near/far CIFAR-100 images with the same
   frozen outputs and thresholds; report AUROC (near / far / all), test
   acceptance, near/far rejection and FPR@95TPR (= fraction of unknowns accepted).

Outputs (results/):
    table1_posthoc_scores_vanilla.{csv,md}   MSP / MLS / Energy / Mahalanobis on Vanilla
    table2_model_comparison.{csv,md}         Vanilla / GCSC / PROSER (MLS) + PROSER placeholder (+ RPL)
    table_all_models_all_scores.csv          every (model, score) pair (appendix)
    fig_scores_vanilla.png                   score distributions + ROC for MSP, MLS, Mahalanobis
    failures_vanilla_mls.{csv,png}           accepted near/far unknowns at the Vanilla MLS threshold
    accepted_unknowns_vanilla_mls.csv        every accepted unknown, most confident first
    per_class_acceptance.csv                 per unknown class: acceptance + absorbing CIFAR-10 label
    absorption_<model>_<score>.csv           unknown class x predicted class counts (accepted only)
    score_agreement_spearman.csv, score_decision_overlap.csv
    summary.md                               all tables + computed facts for the research questions
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.cifar100_unknowns import load_cifar100_unknowns
from evaluation import failure_analysis as fa
from evaluation.metrics import closed_set_accuracy, osr_report, roc
from evaluation.thresholds import calibrate_threshold
from scores import (SCORE_LABELS, MahalanobisScorer, energy_score, mls_score, msp_score,
                    proser_dummy_score, proser_margin_score, rpl_score)
from utils import dump_json, load_config, resolve, sha256_file

MODEL_CONFIGS = {"vanilla": "configs/vanilla.yaml", "gcsc": "configs/gcsc.yaml",
                 "proser": "configs/proser.yaml", "rpl": "configs/rpl.yaml"}
MODEL_LABELS = {"vanilla": "Vanilla", "gcsc": "GCSC", "proser": "PROSER", "rpl": "RPL"}
REQUIRED = ["vanilla", "gcsc", "proser"]
POSTHOC = ["msp", "mls", "energy", "mahalanobis"]
TABLE2_ROWS = [("vanilla", "mls"), ("gcsc", "mls"), ("proser", "mls"),
               ("proser", "proser_dummy"), ("proser", "proser_margin"), ("rpl", "rpl")]
GROUP_COLORS = {"known": "#2a78d6", "near": "#eb6834", "far": "#1baf7a"}  # fixed categorical order
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# --------------------------------------------------------------------------- #
# Loading cached outputs
# --------------------------------------------------------------------------- #
def cache_dir(model: str, overrides) -> Path:
    cfg = load_config(resolve(MODEL_CONFIGS[model]), overrides)
    return resolve(cfg["output"]["cache_dir"])


def load_known(model: str, overrides=()) -> dict:
    z = np.load(cache_dir(model, overrides) / "known.npz")
    out = {}
    for split in ("train", "val", "test"):
        out[split] = {k[len(split) + 1:]: z[k] for k in z.files if k.startswith(split + "_")}
    return out


def load_unknown(model: str, overrides=()) -> dict:
    z = np.load(cache_dir(model, overrides) / "unknown.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def check_cache_fresh(model: str, overrides=()) -> None:
    meta_p = cache_dir(model, overrides) / "meta.json"
    if not meta_p.exists():
        return
    meta = json.load(open(meta_p))
    ck = Path(meta["checkpoint"])
    if ck.exists() and sha256_file(ck) != meta["checkpoint_sha256"]:
        raise SystemExit(f"cache/{model} was extracted from a different checkpoint than {ck}; "
                         f"re-run extract_outputs.py for {model}.")


# --------------------------------------------------------------------------- #
# Scores on frozen outputs
# --------------------------------------------------------------------------- #
class ScoreBank:
    """Computes every u(x) for one model from its saved logits/features."""

    def __init__(self, model: str, known: dict):
        self.model = model
        self.has_dummy = "dummy_logits" in known["val"]
        self._maha = None
        self._train = known["train"]

    @property
    def maha(self) -> MahalanobisScorer:
        if self._maha is None:  # class means + shared diagonal covariance from unaugmented train features
            self._maha = MahalanobisScorer(eps=1e-6).fit(self._train["features"], self._train["labels"])
        return self._maha

    def available(self) -> list[str]:
        if self.model == "rpl":
            return ["rpl", "mahalanobis"]
        return POSTHOC + (["proser_dummy", "proser_margin"] if self.has_dummy else [])

    def __call__(self, score: str, arr: dict) -> np.ndarray:
        z = arr["logits"]
        if score == "msp":
            return msp_score(z)
        if score == "mls":
            return mls_score(z)
        if score == "energy":
            return energy_score(z)
        if score == "mahalanobis":
            return self.maha.score(arr["features"])
        if score == "proser_dummy":
            return proser_dummy_score(z, arr["dummy_logits"])
        if score == "proser_margin":
            return proser_margin_score(z, arr["dummy_logits"])
        if score == "rpl":
            return rpl_score(z)
        raise KeyError(score)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
PCT_COLS = ["csa", "auroc_near", "auroc_far", "auroc_all", "test_accept", "near_reject", "far_reject",
            "all_reject", "fpr95_near", "fpr95_far", "fpr95_all"]
NICE = {"model": "Model", "score": "Score", "csa": "CSA (%)", "auroc_near": "AUROC Near (%)",
        "auroc_far": "AUROC Far (%)", "auroc_all": "AUROC All (%)", "tau": "τ (val 95th pct)",
        "test_accept": "Test accept (%)", "near_reject": "Near reject (%)", "far_reject": "Far reject (%)",
        "all_reject": "All reject (%)", "fpr95_near": "FPR@95 Near (%)", "fpr95_far": "FPR@95 Far (%)",
        "fpr95_all": "FPR@95 All (%)"}


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.4g}" if isinstance(v, float) and c.startswith("τ") else
                         (f"{v:.2f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pretty(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    for c in cols:
        if c in PCT_COLS:
            out[c] = 100.0 * out[c].astype(float)
    if "model" in out:
        out["model"] = out["model"].map(MODEL_LABELS)
    if "score" in out:
        out["score"] = out["score"].map(SCORE_LABELS)
    return out.rename(columns=NICE)


def save_table(df: pd.DataFrame, cols: list[str], stem: Path, title: str) -> str:
    df[cols].to_csv(stem.with_suffix(".csv"), index=False)
    md = f"### {title}\n\n" + to_markdown(pretty(df, cols)) + "\n"
    stem.with_suffix(".md").write_text(md)
    return md


# --------------------------------------------------------------------------- #
# Figure: MSP / MLS / Mahalanobis distributions + ROC (Vanilla)
# --------------------------------------------------------------------------- #
def make_score_figure(u: dict, taus: dict, reports: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold"})
    scores = ["msp", "mls", "mahalanobis"]
    xform = {"msp": (lambda v: np.log10(np.clip(v, 1e-12, None)), "log10 u_MSP = log10(1 − max_k p_k)"),
             "mls": (lambda v: v, "u_MLS = −max_k z_k"),
             "mahalanobis": (lambda v: np.log10(np.clip(v, 1e-12, None)), "log10 u_Mah (min_c distance)")}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    fig.patch.set_facecolor(SURFACE)
    for j, s in enumerate(scores):
        f, xlabel = xform[s]
        ax = axes[0, j]
        ax.set_facecolor(SURFACE)
        vals = {g: f(u[s][g]) for g in ("known", "near", "far")}
        pool = np.concatenate(list(vals.values()))
        lo, hi = np.percentile(pool, [0.5, 99.5])
        if not hi - lo > 1e-9:  # degenerate (e.g. an untrained smoke-test model)
            lo, hi = lo - 0.5, hi + 0.5
        bins = np.linspace(lo, hi, 60)
        names = {"known": "Known (CIFAR-10 test)", "near": "Near unknown", "far": "Far unknown"}
        for g in ("known", "near", "far"):
            v = np.clip(vals[g], lo, hi)
            ax.hist(v, bins=bins, density=True, histtype="stepfilled", alpha=0.14, color=GROUP_COLORS[g])
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.8, color=GROUP_COLORS[g], label=names[g])
        t = float(f(np.array([taus[s]]))[0])
        ax.axvline(t, color=INK2, lw=1.2, ls="--")
        ax.text(t, ax.get_ylim()[1] * 0.97, " τ (val 95%)", color=INK2, fontsize=8, va="top", ha="left")
        ax.set_title(SCORE_LABELS[s])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density" if j == 0 else "")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)

        ax = axes[1, j]
        ax.set_facecolor(SURFACE)
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
        rep = reports[s]
        for g in ("near", "far"):
            fpr, tpr = roc(u[s]["known"], u[s][g])
            ax.plot(fpr, tpr, color=GROUP_COLORS[g], lw=2,
                    label=f"Known vs {g.capitalize()}  AUROC {100 * rep[f'auroc_{g}']:.1f}")
            ax.plot([1 - rep["test_accept"]], [rep[f"{g}_reject"]], "o", ms=7, color=GROUP_COLORS[g],
                    mec=SURFACE, mew=2)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.01)
        ax.set_xlabel("known test rejected (FPR)")
        ax.set_ylabel("unknown rejected (TPR)" if j == 0 else "")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    fig.suptitle("Vanilla ResNet-18: score distributions (top) and ROC curves (bottom); "
                 "dots = validation-calibrated operating point", color=INK, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Failure analysis (Vanilla + MLS threshold by default)
# --------------------------------------------------------------------------- #
def run_failure_analysis(model="vanilla", score="mls", per_group=8, results_dir="results", overrides=()):
    res = resolve(results_dir)
    res.mkdir(parents=True, exist_ok=True)
    known, unk = load_known(model, overrides), load_unknown(model, overrides)
    bank = ScoreBank(model, known)
    tau = calibrate_threshold(bank(score, known["val"]))
    u = bank(score, unk)
    pred = unk["logits"].argmax(1)
    conf = 1.0 - msp_score(unk["logits"])
    acc_df = fa.accepted_unknowns(unk, u, tau, pred, conf)
    acc_df.to_csv(res / f"accepted_unknowns_{model}_{score}.csv", index=False)
    near, far = fa.select_failures(acc_df, "near", per_group), fa.select_failures(acc_df, "far", per_group)
    pd.concat([near, far]).to_csv(res / f"failures_{model}_{score}.csv", index=False)

    cfg = load_config(resolve(MODEL_CONFIGS[model]), overrides)
    d = cfg["data"]
    raw = load_cifar100_unknowns(d.get("root", "data/raw"), download=d.get("download", True),
                                 synthetic=d.get("synthetic"))
    if not np.array_equal(raw["cifar100_index"], unk["cifar100_index"]):
        raise RuntimeError("CIFAR-100 image order does not match the cached unknown outputs")
    fa.plot_failure_grid(raw["images"], near, far, res / f"failures_{model}_{score}.png",
                         f"Accepted unknowns – {MODEL_LABELS[model]} + {SCORE_LABELS[score]}")
    print(f"[failures] {len(acc_df)} unknowns accepted by {model}/{score} "
          f"(near {int((acc_df.group == 'near').sum())}, far {int((acc_df.group == 'far').sum())}); "
          f"figure -> {res / f'failures_{model}_{score}.png'}")
    return acc_df, near, far, tau


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--failures-per-group", type=int, default=8)
    ap.add_argument("--override", nargs="*", default=[], help="applied to every model config (e.g. data.*)")
    args = ap.parse_args()
    ov = args.override
    res = resolve(args.results_dir)
    res.mkdir(parents=True, exist_ok=True)

    models = args.models
    if models is None:
        models = list(REQUIRED)
        if (cache_dir("rpl", ov) / "unknown.npz").exists():
            models.append("rpl")
    for m in models:
        for f in ("known.npz", "unknown.npz"):
            if not (cache_dir(m, ov) / f).exists():
                raise SystemExit(f"Missing {cache_dir(m, ov) / f}; run extract_outputs.py for {m} first.")
        check_cache_fresh(m, ov)

    # ---------------- Stage 1: known data only (CSA + thresholds) --------------
    banks, known_u, calib, csa = {}, {}, {}, {}
    for m in models:
        known = load_known(m, ov)
        banks[m] = ScoreBank(m, known)
        csa[m] = closed_set_accuracy(known["test"]["logits"], known["test"]["labels"])
        for s in banks[m].available():
            u_val, u_test = banks[m](s, known["val"]), banks[m](s, known["test"])
            tau = calibrate_threshold(u_val)
            known_u[(m, s)] = (u_val, u_test)
            calib[f"{m}/{s}"] = {"tau": tau, "val_accept": float(np.mean(u_val <= tau)),
                                 "test_accept": float(np.mean(u_test <= tau)), "csa": csa[m]}
    dump_json({"rule": "tau = 95th percentile of u on CIFAR-10 validation; accept iff u <= tau",
               "thresholds": calib}, res / "thresholds.json")
    print("[eval] CSA: " + ", ".join(f"{MODEL_LABELS[m]} {100 * csa[m]:.2f}%" for m in models))

    # ---------------- Stage 2: unknowns ----------------------------------------
    rows, unk_u, unk_meta, preds = [], {}, {}, {}
    for m in models:
        unk = load_unknown(m, ov)
        unk_meta[m] = unk
        preds[m] = unk["logits"].argmax(1)
        near, far = unk["group"] == "near", unk["group"] == "far"
        for s in banks[m].available():
            u = banks[m](s, unk)
            unk_u[(m, s)] = u
            u_val, u_test = known_u[(m, s)]
            rep = osr_report(u_val, u_test, u[near], u[far], calib[f"{m}/{s}"]["tau"])
            rows.append({"model": m, "score": s, "csa": csa[m], **rep})
    all_df = pd.DataFrame(rows)
    all_df.to_csv(res / "table_all_models_all_scores.csv", index=False)

    metric_cols = ["auroc_near", "auroc_far", "auroc_all", "tau", "test_accept", "near_reject",
                   "far_reject", "all_reject", "fpr95_near", "fpr95_far", "fpr95_all"]
    t1 = all_df[(all_df.model == "vanilla") & all_df.score.isin(POSTHOC)].copy()
    t1["_o"] = t1.score.map(POSTHOC.index)
    t1 = t1.sort_values("_o").drop(columns="_o")
    md1 = save_table(t1, ["score"] + metric_cols, res / "table1_posthoc_scores_vanilla",
                     "Table 1 – Post-hoc scores on the frozen Vanilla model")
    t2 = pd.concat([all_df[(all_df.model == m) & (all_df.score == s)] for m, s in TABLE2_ROWS
                    if m in models and ((all_df.model == m) & (all_df.score == s)).any()])
    md2 = save_table(t2, ["model", "score", "csa"] + metric_cols, res / "table2_model_comparison",
                     "Table 2 – Trained-model comparison (MLS common score + PROSER placeholder score)")
    print("\n" + md1 + "\n" + md2)

    # ---------------- Figure ---------------------------------------------------
    vn = unk_meta["vanilla"]["group"]
    u_fig = {s: {"known": known_u[("vanilla", s)][1], "near": unk_u[("vanilla", s)][vn == "near"],
                 "far": unk_u[("vanilla", s)][vn == "far"]} for s in ("msp", "mls", "mahalanobis")}
    reps = {s: t1[t1.score == s].iloc[0].to_dict() for s in ("msp", "mls", "mahalanobis")}
    make_score_figure(u_fig, {s: calib[f"vanilla/{s}"]["tau"] for s in u_fig}, reps, res / "fig_scores_vanilla.png")

    # ---------------- Score agreement (Vanilla) -------------------------------
    pool = {s: np.r_[known_u[("vanilla", s)][1], unk_u[("vanilla", s)]] for s in POSTHOC}
    unk_only = {s: unk_u[("vanilla", s)] for s in POSTHOC}
    sp_rows = []
    for a, b in itertools.combinations(POSTHOC, 2):
        sp_rows.append({"score_a": a, "score_b": b,
                        "spearman_test_plus_unknown": spearmanr(pool[a], pool[b])[0],
                        "spearman_known_test": spearmanr(known_u[("vanilla", a)][1], known_u[("vanilla", b)][1])[0],
                        "spearman_unknown_only": spearmanr(unk_only[a], unk_only[b])[0]})
    sp_df = pd.DataFrame(sp_rows)
    sp_df.to_csv(res / "score_agreement_spearman.csv", index=False)
    ov_rows = []
    rej = {s: unk_u[("vanilla", s)] > calib[f"vanilla/{s}"]["tau"] for s in POSTHOC}
    for a, b in itertools.combinations(POSTHOC, 2):
        for g in ("near", "far"):
            m_ = vn == g
            ra, rb = rej[a][m_], rej[b][m_]
            ov_rows.append({"score_a": a, "score_b": b, "group": g, "both_reject": float(np.mean(ra & rb)),
                            "only_a_rejects": float(np.mean(ra & ~rb)), "only_b_rejects": float(np.mean(~ra & rb)),
                            "both_accept": float(np.mean(~ra & ~rb))})
    ov_df = pd.DataFrame(ov_rows)
    ov_df.to_csv(res / "score_decision_overlap.csv", index=False)

    # ---------------- Per-class acceptance / absorption ------------------------
    pc = []
    for (m, s) in [("vanilla", x) for x in POSTHOC] + [r for r in TABLE2_ROWS if r[0] != "vanilla"]:
        if (m, s) not in unk_u:
            continue
        tau = calib[f"{m}/{s}"]["tau"]
        d = fa.per_class_acceptance(unk_meta[m], unk_u[(m, s)], tau, preds[m])
        d.insert(0, "score", s)
        d.insert(0, "model", m)
        pc.append(d)
        if s in ("mls", "proser_dummy", "rpl"):
            fa.absorption_matrix(unk_meta[m], unk_u[(m, s)], tau, preds[m]).to_csv(res / f"absorption_{m}_{s}.csv")
    pc_df = pd.concat(pc)
    pc_df.to_csv(res / "per_class_acceptance.csv", index=False)

    # ---------------- Failure analysis (Vanilla MLS threshold) -----------------
    acc_df, fail_near, fail_far, tau_v = run_failure_analysis("vanilla", "mls", args.failures_per_group,
                                                              args.results_dir, ov)

    # ---------------- Summary --------------------------------------------------
    write_summary(res, models, csa, all_df, md1, md2, sp_df, ov_df, pc_df, fail_near, fail_far, tau_v)
    print(f"[eval] wrote results to {res}")


def write_summary(res, models, csa, all_df, md1, md2, sp_df, ov_df, pc_df, fail_near, fail_far, tau_v):
    def row(m, s):
        r = all_df[(all_df.model == m) & (all_df.score == s)]
        return None if r.empty else r.iloc[0]

    def pp(x):
        return f"{100 * x:.2f}"

    L = ["# Task 4 – Open-Set Recognition: results summary", "",
         "_Auto-generated by `evaluate_osr.py`. Numbers are computed from the frozen caches; the "
         "interpretation is yours to write. Thresholds were fixed on CIFAR-10 validation data "
         "before any CIFAR-100 image was scored (`thresholds.json`)._", "",
         "## Closed-set accuracy (CIFAR-10 test, 10 known logits)", ""]
    L += [f"- {MODEL_LABELS[m]}: **{pp(csa[m])} %**" for m in models]
    L += ["", md1, "", md2, "", "![scores](fig_scores_vanilla.png)", ""]

    # RQ1
    v = pc_df[(pc_df.model == "vanilla") & (pc_df.score == "mls")]
    L += ["## RQ1 – Semantic similarity (Vanilla + MLS threshold)", "",
          "| Group | Unknown class | Accept rate (%) | Top absorbing CIFAR-10 label | Share of accepted (%) |",
          "|---|---|---|---|---|"]
    for _, r in v.iterrows():
        L.append(f"| {r.group} | {r.unknown_class} | {pp(r.accept_rate)} | {r.top_absorbing_class} | "
                 f"{pp(r.top_absorbing_share_of_accepted)} |")
    L += ["", f"Failure gallery (τ = {tau_v:.4f}): `failures_vanilla_mls.png`, details in "
          "`failures_vanilla_mls.csv` (fill the `your_judgement` column after looking at the images).", "",
          "| Group | Unknown class | Predicted | u | τ | a-priori label |", "|---|---|---|---|---|---|"]
    for df in (fail_near, fail_far):
        for _, r in df.iterrows():
            L.append(f"| {r.group} | {r.unknown_class} | {r.predicted_class} | {r.score_u:.4f} | "
                     f"{r.threshold_tau:.4f} | {r.a_priori_semantic_neighbour} |")

    # RQ2
    L += ["", "## RQ2 – What each post-hoc score captures (Vanilla)", "",
          "Spearman rank correlation of u(x):", "",
          "| Pair | Known test + unknowns | Known test only | Unknowns only |", "|---|---|---|---|"]
    for _, r in sp_df.iterrows():
        L.append(f"| {SCORE_LABELS[r.score_a]} vs {SCORE_LABELS[r.score_b]} | {r.spearman_test_plus_unknown:.3f} | "
                 f"{r.spearman_known_test:.3f} | {r.spearman_unknown_only:.3f} |")
    L += ["", "Decision disagreement on unknowns at each score's own validation threshold "
          "(% of the group rejected by one score but not the other):", "",
          "| Pair | Group | Only A rejects | Only B rejects | Both reject |", "|---|---|---|---|---|"]
    for _, r in ov_df.iterrows():
        L.append(f"| A={SCORE_LABELS[r.score_a]}, B={SCORE_LABELS[r.score_b]} | {r.group} | "
                 f"{pp(r.only_a_rejects)} | {pp(r.only_b_rejects)} | {pp(r.both_reject)} |")

    # RQ3 / RQ4 deltas
    L += ["", "## RQ3 / RQ4 – Deltas relative to Vanilla + MLS (percentage points)", "",
          "| Model / score | ΔCSA | ΔAUROC Near | ΔAUROC Far | ΔNear reject | ΔFar reject |", "|---|---|---|---|---|---|"]
    base = row("vanilla", "mls")
    for m, s in TABLE2_ROWS[1:]:
        r = row(m, s)
        if r is None:
            continue
        L.append(f"| {MODEL_LABELS[m]} + {SCORE_LABELS[s]} | {100 * (r.csa - base.csa):+.2f} | "
                 f"{100 * (r.auroc_near - base.auroc_near):+.2f} | {100 * (r.auroc_far - base.auroc_far):+.2f} | "
                 f"{100 * (r.near_reject - base.near_reject):+.2f} | {100 * (r.far_reject - base.far_reject):+.2f} |")
    g, p = row("gcsc", "mls"), row("proser", "proser_dummy")
    if g is not None and p is not None:
        L += ["", f"PROSER (placeholder score) vs GCSC (MLS): ΔCSA {100 * (p.csa - g.csa):+.2f}, "
              f"ΔAUROC near {100 * (p.auroc_near - g.auroc_near):+.2f}, "
              f"ΔAUROC far {100 * (p.auroc_far - g.auroc_far):+.2f}."]
    L += ["", "## Questions to answer in the report", "",
          "1. How does semantic similarity affect rejection? Which near/far classes are most often accepted, "
          "which CIFAR-10 labels absorb them, which failures are semantically plausible?",
          "2. What do MSP, MLS, Energy and Mahalanobis each capture? Use their agreements/disagreements.",
          "3. Does GCSC's RandAugment improve CSA, OSR, both or neither? Why do near and far differ?",
          "4. Does PROSER improve rejection beyond Vanilla and GCSC, at what CSA cost? Where is "
          "interpolation between known classes insufficient?", ""]
    (res / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
