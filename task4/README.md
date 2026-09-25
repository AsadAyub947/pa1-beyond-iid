# Task 4 – Open-Set Recognition (OSR)

CIFAR-10 (10 known classes) vs. fixed CIFAR-100 **test** classes as unknowns:

| Group | CIFAR-100 fine classes (800 images each group) |
|---|---|
| Near unknown | bus, pickup_truck, motorcycle, tractor, wolf, fox, leopard, camel |
| Far unknown | bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe |

Methods: **Vanilla** ResNet-18 + post-hoc scores (MSP, MLS, Energy, Mahalanobis),
**GCSC** (Vanilla + RandAugment, scored with MLS), **PROSER** (classifier + data
placeholders), and the optional **RPL** extension.

Dataset construction, training, score extraction and evaluation are separate
scripts, so every score for a model is computed from **one** saved set of
logits/features.

---

## 2. Setup

```bash
cd task4
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# For a specific CUDA build of PyTorch, install torch/torchvision first from https://pytorch.org
```
---

## 3. Run it – step by step

Run every command from inside `task4/`.

### Step 0 – Data and the fixed split
```bash
python -m data.make_splits
```
Downloads CIFAR-10 and CIFAR-100 to `data/raw/` and writes the stratified 90/10
split of the official CIFAR-10 training set (seed 6304) to
`data/splits/cifar10_train90_val10_seed6304.json` (45 000 train / 5 000 val, 4 500 / 500 per class).
Only the CIFAR-100 *test* partition is ever read.

### Step 1 – Vanilla closed-set baseline
```bash
python train.py --config configs/vanilla.yaml
```
ResNet-18 (3×3 stride-1 stem, no max-pool), cross-entropy, crop(32, pad 4) + flip,
SGD 0.1 / momentum 0.9 / wd 5e-4, cosine, batch 128, 100 epochs, seed 6304.
`checkpoints/vanilla/best.pt` = highest CIFAR-10 validation accuracy.

### Step 2 – Post-hoc novelty scores
Nothing to train: MSP, MLS, Energy and Mahalanobis are computed in Step 6 from
the frozen Vanilla outputs.

### Step 3 – GCSC (RandAugment + MLS)
```bash
python train.py --config configs/gcsc.yaml
```
Same recipe with exactly one change: `RandAugment(num_ops=2, magnitude=9)` after crop+flip, before ToTensor/Normalize.

### Step 4 – PROSER
```bash
python train.py --config configs/proser.yaml      # needs checkpoints/vanilla/best.pt
```
Starts from the selected Vanilla checkpoint + 5 random dummy classifiers; fine-tunes
the whole network for 50 epochs (SGD 1e-3, cosine, batch 128, β = 1, γ = 0.1, mixup after layer2, λ ~ Beta(2,2)).

### Step 5 – Optional: RPL
```bash
python train.py --config configs/rpl.yaml
```

### Step 6 – Freeze models and cache outputs
```bash
python extract_outputs.py --config configs/vanilla.yaml configs/gcsc.yaml configs/proser.yaml
python extract_outputs.py --config configs/rpl.yaml           # only if you trained RPL
```
Writes `cache/<model>/known.npz` (CIFAR-10 train-unaugmented / val / test) and
`cache/<model>/unknown.npz` (1 600 CIFAR-100 images). The script **refuses** to
touch unknowns until the Vanilla, GCSC and PROSER checkpoints all exist, then records
their SHA-256 hashes in `results/frozen_manifest.json`. If a checkpoint changes afterwards,
extraction and evaluation stop with an error (the protocol forbids revising models after seeing unknowns).

### Step 7 – Evaluate
```bash
python evaluate_osr.py              # adds RPL automatically if cache/rpl/unknown.npz exists
```
First computes CSA and every threshold τ from CIFAR-10 **validation** data only
(written to `results/thresholds.json`), then scores the unknowns.

### Or everything at once
```bash
bash run_all.sh                     # required methods
WITH_RPL=1 bash run_all.sh          # + optional RPL
```

**Interrupted run?** Add `--resume` to the same `train.py` command; it continues from
`checkpoints/<model>/last.pt` (model, optimiser, scheduler, AMP scaler and RNG state).

---

## 4. Where each piece of required evidence ends up (`results/`)

| Required evidence | File(s) |
|---|---|
| Table: MSP / MLS / Energy / Mahalanobis on frozen Vanilla (near / far / all AUROC + val-calibrated rejection) | `table1_posthoc_scores_vanilla.md/.csv` |
| Table: Vanilla vs GCSC vs PROSER (MLS) + PROSER placeholder score, with CSA and near/far OSR | `table2_model_comparison.md/.csv` (RPL row added if trained) |
| Compact score-distribution + ROC figure for MSP, MLS, Mahalanobis | `fig_scores_vanilla.png` |
| ≥ 3 near + ≥ 3 far accepted unknowns (class, predicted label, score, threshold) | `failures_vanilla_mls.png`, `failures_vanilla_mls.csv` (8 per group; one per class first) |
| RQ1 – which classes are accepted, which labels absorb them | `per_class_acceptance.csv`, `absorption_vanilla_mls.csv`, `accepted_unknowns_vanilla_mls.csv` |
| RQ2 – agreement / disagreement between scores | `score_agreement_spearman.csv`, `score_decision_overlap.csv` |
| RQ3 / RQ4 – CSA vs OSR deltas | `summary.md` (delta table), `table_all_models_all_scores.csv` |
| Everything in one place | `summary.md` |

Every table reports AUROC for Known vs Near, Known vs Far, Known vs All; τ; CIFAR-10
test acceptance; near / far / all rejection; and FPR@95TPR (= fraction of unknowns
accepted at τ). Percentages are ×100 in the `.md` tables, raw fractions in the `.csv` files.

---

## 5. Implementation details

**Split / data.** `sklearn.model_selection.train_test_split(stratify=labels, test_size=0.1, random_state=6304)`
on the official 50 000 training images. Validation is used for checkpoint selection and
thresholds only. CIFAR-10 normalisation is applied to all inputs (including CIFAR-100).

**Model.** `torchvision.models.resnet18(weights=None)` with `conv1 = Conv2d(3, 64, 3, stride 1, pad 1)`
and `maxpool = Identity`. Penultimate feature f(x) = 512-d global-average-pooled output of layer4.

**Training.** Cosine annealing over the epoch budget, stepped once per epoch; weight decay on all
parameters; mixed precision on CUDA (losses computed in float32). Data order and augmentation are
re-seeded every epoch from seed 6304, so `--resume` reproduces an uninterrupted run.

**Scores** (all: larger = more novel, same saved arrays):
- `u_MSP = 1 − max_k softmax(z)_k`
- `u_MLS = −max_k z_k`
- `u_Energy = −log Σ_k exp(z_k)` (T = 1)
- `u_Mah = min_c Σ_j (f_j − μ_cj)² / σ_j²`; μ_c = class means, σ² = pooled within-class variance per
  dimension (one shared diagonal Σ) from **unaugmented** CIFAR-10 training features, + 1e-6 on the diagonal.

**Threshold.** τ = 95th percentile of u on CIFAR-10 validation; accept iff u ≤ τ.

**PROSER** (`methods/proser.py`, following Zhou et al. 2021):
- Output f̂(x) = [z₁…z₁₀, max_c d_c(x)] – the strongest of the 5 dummy classifiers is the K+1 "unknown" slot.
- First half of each batch (64 images), classifier placeholders:
  `CE(f̂(x), y) + β·CE(f̂(x)\y, K+1)` where `f̂(x)\y` sets the ground-truth logit to −1e9, so the dummy
  must be the strongest response once the true class is excluded (β = 1).
- Second half (64 images), data placeholders: `h̃ = λ·φ_pre(x_i) + (1−λ)·φ_pre(x_j)` after layer2, with each
  partner j drawn uniformly among batch images of a **different** class, λ ~ Beta(2,2) (one λ per batch as in
  the reference code; `lambda_per_sample: true` draws one per pair); `h̃` goes through layer3→layer4→pool→heads
  and is trained toward the K+1 slot: `γ·CE(f̂(h̃), K+1)`, γ = 0.1.
- Total `L = l₁ + γ·l₂`. Checkpoint = best CIFAR-10 val accuracy of the 10 known logits (CSA also uses only those).
- Detection score (reference code): `u = softmax([z₁…z₁₀, max_c d_c])₁₁`, calibrated at the 95th val percentile.
  The paper's calibration (add a bias to the dummy logit until 95 % of val data stays known) is identical to
  thresholding `max_c d_c − max_k z_k`; that is reported as an extra row ("dummy margin").
- Differences from the public reference code, all following the task text: the task's half assignment (first half =
  classifier placeholders, second half = mixup; the reference does the reverse), mixing only different-class pairs
  (the reference mixes a random permutation), and the paper's K+1 formulation (max over dummies) in every loss term.

---