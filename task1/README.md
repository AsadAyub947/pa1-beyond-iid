# Task 1: Inductive Biases and Feature Representations

## Design choices made

- **Dataset:** STL-10 (the assignment's recommended option).
- **Additional color intervention:** fixed hue rotation (90°, i.e. `hue_shift=0.25`
  in `[-0.5, 0.5]`) rather than palette transfer or class-swapped stats — simplest
  to reason about and to control the "what does it change vs. preserve" question.
- **AdaIN implementation:** *optimization-based* AdaIN (a frozen VGG-19 encoder +
  per-image pixel optimization to match AdaIN-transformed features), not the
  original feed-forward decoder network, since no pretrained AdaIN decoder ships
  with torchvision/open_clip. This is slower (~300 optimizer steps/image,
  configurable) but requires no extra downloaded weights.
- **Cue-conflict source images:** drawn from the STL-10 **test** partition,
  excluding the 500 images already selected for the main evaluation subset.
  (The **train** partition is fully consumed by the 80/20 split used to train
  the linear heads, so it isn't a usable source pool.)
- **Representation-stability translation displacement:** the largest
  displacement (32px) is used for the t-SNE/UMAP plot (most likely to show a
  visible effect); all displacements are reported numerically in
  `task1_results.json`.

## Setup

```bash
# 1. Create environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate         # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # or cpu-only URL
pip install open_clip_torch scikit-learn matplotlib pyyaml pillow numpy
pip install umap-learn          # only needed if you set representation.method: umap in config
```

## Running the pipeline

Run these **in order**, from inside `task1/`:

```bash
cd task1

# Step 1: build the stratified train/val split + class-balanced 500-image test subset
python data/make_subset.py --config configs/config.yaml

# Step 2: generate the cue-conflict (shape/texture) images via AdaIN
python data/make_cue_conflicts.py --config configs/config.yaml

# Step 3: run everything else -- feature extraction, linear-head training,
python scripts/run_task1.py --config configs/config.yaml
```

## Outputs

After a full run, `results/` contains:
- `subset_indices.json` — the exact image indices used everywhere (for reproducibility)
- `cue_conflicts/` — generated stylized images (`.pt` tensors) + `metadata.json`
  (content/style labels, accepted/rejected counts per the visual rejection rule)
- `task1_results.json` — every required metric: clean baseline, color-bias
  report, shape-bias/coverage, translation curve (per displacement, averaged
  over the 4 directions), patch-shuffle accuracy drop, representation-stability
  (cosine `I_T`) for grayscale/hue/cue-conflict/patch-shuffle/each translation
  displacement, and the CLIP zero-shot vs. trained-head agreement — for all
  three backbones (plus CLIP zero-shot separately)
- `plots/<backbone_name>/*.png` — t-SNE (or UMAP) visualizations of clean vs.
  each transformed condition, colored by class and marked by condition