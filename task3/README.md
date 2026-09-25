# Task 3: Domain Generalization (PACS, Sketch as unseen target)

## Prerequisites

**Task 2 must be run first.** This task loads `task2/results/checkpoints/source_only.pt`
directly as its ERM baseline (per the assignment: "load its saved checkpoint
rather than retraining it"). It also reuses Task 2's cached
`shared/splits/pacs_sketch_seed6304.json` automatically (same `shared/`
package, same seed) — nothing to configure, just don't regenerate that file
independently from Task 3.

Expected sibling layout:
```
repo_root/
    shared/                 (from Task 2)
    task2/                  (from Task 2 -- source_only.pt must already exist)
    task3/                  (this deliverable)
```

## Setup

Same as Task 2:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu-only URL
pip install scikit-learn pyyaml pillow numpy
```

## Running the pipeline

```bash
cd task3

# Train DAN-DG and SAM (ERM is NOT trained here -- see configs/erm.yaml):
python train.py --config configs/dan_dg.yaml
python train.py --config configs/sam.yaml

# Final evaluation -- the ONLY point Sketch is ever loaded. Loads Task 2's
# source_only.pt as ERM, evaluates all 3 methods on source-val + Sketch,
# computes source-domain separability and the sharpness proxy for all 3:
python evaluate_sketch.py --base_config configs/base.yaml

# Controlled design study (Step 5) -- choose ONE:
python evaluate_sketch.py --base_config configs/base.yaml --controlled_study dan_dg
#   sweeps lambda_dg in {0.1, 1, 10}
# OR:
python evaluate_sketch.py --base_config configs/base.yaml --controlled_study sam
#   sweeps rho in {0.01, 0.05, 0.1}
```

Note `train.py` will refuse to run `configs/erm.yaml` with a clear error
message, by design — this structurally prevents accidentally retraining the
ERM baseline under different settings than Task 2 used.

## Outputs

- `results/checkpoints/{dan_dg,sam}.pt` — best checkpoint by mean
  source-validation macro-F1
- `results/{dan_dg,sam}_history.json` — per-epoch training curves, including
  the MMD penalty for DAN-DG and both loss values (clean + perturbed-point)
  for SAM (Required Evidence: training/alignment curves)
- `results/final_report.json` — comparison table (mean/worst-source, Sketch,
  Sketch change vs. ERM, source-domain separability, sharpness), per-class
  Sketch deltas/confusions vs. ERM, and — if Task 2's `final_report.json` is
  present — a direct pull-through of Task 2's DAN numbers for the
  target-aware-vs-target-free comparison Research Question 4 asks for
- `results/controlled_study_{dan_dg,sam}.json` — the controlled design study
  table (Required Evidence)

## Design choices made

- **ERM's "domain-balanced" pooling:** implemented as plain pooled
  cross-entropy over equal-sized per-domain batches (8 each) — mathematically
  equivalent to `(1/3)∑R_e` in expectation without needing three separate
  loss terms.
- **SAM optimizer:** implemented as an explicit two-method wrapper
  (`ascent_step()` / `restore_and_step()`) around `AdamW`, rather than the
  `optimizer.step(closure=...)` pattern some public SAM implementations use —
  functionally identical, but keeps the two-pass control flow visible in
  `SAMMethod.step()` rather than hidden inside a closure.
- **Sharpness diagnostic scope:** computed using only source-validation data
  (as specified), so it — like source-domain separability — is technically
  computable before Sketch is touched. It's implemented inside
  `evaluate_sketch.py` anyway since the assignment groups it under one
  combined "Common Evaluation and Diagnostics" step; this doesn't violate the
  no-target-leakage rule since checkpoints are already frozen by then.