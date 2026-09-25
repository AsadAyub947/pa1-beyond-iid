# Task 2: Unsupervised Domain Adaptation (PACS, Sketch as target)

## PACS

Arrange it as:
```
pacs_data/
    photo/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
    art_painting/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
    cartoon/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
    sketch/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg
```
and point `pacs.root` in `task2/configs/base.yaml` at it.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 
pip install scikit-learn pyyaml pillow numpy
```
## Running the pipeline

From inside `task2/` (paths in the configs are relative to this directory):

```bash
cd task2

# Train each method (order doesn't matter, but Source-only should be trained
# first since its checkpoint is the ERM baseline referenced everywhere):
python train.py --config configs/source_only.yaml
python train.py --config configs/dan.yaml
python train.py --config configs/dann.yaml
python train.py --config configs/cdan.yaml

# Final evaluation: loads all 4 checkpoints, evaluates on source-val AND
# (for the first time) target labels, builds the comparison table, domain
# separability, and per-class analysis:
python evaluate_final.py --base_config configs/base.yaml

# Controlled design study (Step 6) -- choose ONE:
python evaluate_final.py --base_config configs/base.yaml --controlled_study dan
#   sweeps lambda_mmd in {0.1, 1, 10}, retrains DAN 3x, evaluates each
# OR:
python evaluate_final.py --base_config configs/base.yaml --controlled_study dann
#   sweeps max_alpha (GRL reversal strength) in {0.25, 0.5, 1}, retrains DANN 3x
```

The first `--controlled_study` invocation will re-run `evaluate_final`'s main
comparison too; if you've already generated `final_report.json` and only want
the controlled study, that's fine — it's independent and writes to its own
`controlled_study_<method>.json`.

## Outputs

- `results/checkpoints/{source_only,dan,dann,cdan}.pt` — best checkpoint
  (by mean source-validation macro-F1) for each method
- `results/{method}_history.json` — per-epoch training/validation curves
  (Required Evidence: "classification and alignment or domain-loss curves")
- `results/final_report.json` — the full comparison table, per-class
  deltas/confusions vs. Source-only, and domain separability for all 4
  methods (Required Evidence #1–3)
- `results/controlled_study_{dan,dann}.json` — the controlled design study
  table (Required Evidence #4)
- `shared/splits/pacs_sketch_seed6304.json` — the exact stratified splits
  used, shared verbatim with Task 3 as required

## Design choices

- **Domain-balanced batch composition for Source-only:** uses the same 8+8+8
  domain-balanced source sampling as the adaptation methods (rather than a
  single pooled/shuffled source loader), per "keep ... source sampling ...
  fixed across methods."
- **"Epoch" length:** defined as `max(len(loader) for loader in the three
  source train loaders)` — i.e., one full pass through whichever source
  domain has the most training images, with the other two (and the target
  loader) cycling as needed to fill out that many steps.
- **MMD median-heuristic bandwidth:** computed from the off-diagonal (i.e.,
  excluding self-distances) pairwise squared distances in the current
  combined source+target batch, recomputed every step.
- **Controlled study:** the code supports both options (DAN's λ_MMD or
  DANN's max reversal strength) via `--controlled_study {dan,dann}`; the
  assignment asks you to choose **one** — pick and report on just one in your
  writeup even though both are implemented.