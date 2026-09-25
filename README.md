# PA1 – Learning Beyond IID

Code, configurations, split indices and machine-readable results for Programming Assignment 1
(EE-5102/CS-6304, Fall 2026):

| Task | Topic | Data | Methods |
|---|---|---|---|
| 1 | Inductive biases and feature representations | STL-10 | ResNet-50, ViT-B/16, CLIP ViT-B/32 (linear heads + zero-shot) |
| 2 | Unsupervised domain adaptation | PACS (Sketch = target) | Source-only, DAN, DANN, CDAN |
| 3 | Domain generalization | PACS (Sketch unseen) | ERM (= Task 2 Source-only), DAN-DG, SAM |
| 4 | Open-set recognition | CIFAR-10 known / CIFAR-100 unknown | Vanilla + MSP/MLS/Energy/Mahalanobis, GCSC, PROSER |

Report: [`report/report.pdf`](report/report.pdf)

## Repository layout

```
pa1-beyond-iid/
  README.md  requirements.txt  .gitignore
  shared/                 code genuinely shared by Tasks 2 and 3 (plays the role of common/)
    pacs.py  pacs_protocol.py  download_pacs.py  mmd.py  seed.py
    splits/pacs_sketch_seed6304.json      PACS source train/val split (seed 6304), reused by Task 3
  task1/                  configs/ data/ models/ analysis/ scripts/ results/ task1.ipynb
  task2/                  configs/ models/ methods/ evaluation/ train.py run_controlled_study.py
                          evaluate_final.py results/ task2.ipynb
  task3/                  configs/ models/ methods/ selection/ evaluation/ train.py
                          run_controlled_study.py evaluate_sketch.py results/ task3.ipynb
  task4/                  configs/ data/ models/ methods/ scores/ evaluation/ train.py
                          extract_outputs.py evaluate_osr.py utils.py results/ task4.ipynb
  report/                 report.pdf, figures/ (copies of the figures used in the report)
```

Each `taskN/README.md` documents that task's commands, outputs, design choices and attributions.
Each `taskN/taskN.ipynb` is the executed Colab notebook that produced the committed results; the cell outputs are its run log.
The only cross-task utilities are the PACS protocol, the seeding helpers and the MMD used by both DAN and DAN-DG, so they live in `shared/`. That is the name the assignment's Task 2/3 structure uses for them.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate   # or use Google Colab (T4 GPU)
pip install -r requirements.txt
```
All results were produced on Google Colab with a T4 GPU. Every experiment uses seed 6304.
Tasks 2 and 3 run with deterministic cuDNN and seeded data loaders.

## Data (not committed; downloaded by the code)

| Dataset | How it is obtained |
|---|---|
| STL-10 | `torchvision.datasets.STL10(download=True)` into `task1/data_cache/` |
| PACS | `python shared/download_pacs.py --out /content/pacs_data`: clones <https://github.com/MachineLearning2020/Homework3-PACS>, moves the original JPEGs without re-encoding, and verifies all 9,991 images (per domain and class) |
| CIFAR-10 / CIFAR-100 | `torchvision` download inside `task4/data/make_splits.py` into `task4/data/raw/` (only the CIFAR-100 **test** split is used) |
| AdaIN weights (Task 1) | `decoder.pth` and `vgg_normalised.pth` from <https://github.com/naoto0804/pytorch-AdaIN/releases>, downloaded automatically into `task1/weights/` |

## Reproducing the results

Run from inside each task folder. On Colab, open the task's notebook. The notebooks expect this repository at `/content/pa1`, for example via
`!git clone https://github.com/<AsadAyub947>/pa1-beyond-iid.git /content/pa1`
(the Task 4 notebook uses `/content/task4`).

**Task 1**
```bash
cd task1
python data/make_subset.py --config configs/config.yaml
python data/make_cue_conflicts.py --config configs/config.yaml
python scripts/run_task1.py --config configs/config.yaml
```

**Task 2** (must run before Task 3)
```bash
cd task2
python ../shared/download_pacs.py --out /content/pacs_data
for c in source_only dan dann cdan; do python train.py --config configs/$c.yaml; done
python train.py --config configs/dann_stabilized.yaml   # supplementary: main DANN diverged
python train.py --config configs/cdan_stabilized.yaml   # supplementary: main CDAN diverged
python run_controlled_study.py --study dan
python evaluate_final.py --controlled_study dan
```

**Task 3** (uses Task 2's split file and `task2/results/checkpoints/source_only.pt`)
```bash
cd task3
python train.py --config configs/dan_dg.yaml
python train.py --config configs/sam.yaml
python run_controlled_study.py --study dan_dg
python evaluate_sketch.py --controlled_study dan_dg
```

**Task 4**
```bash
cd task4
python -m data.make_splits
for c in vanilla gcsc proser; do python train.py --config configs/$c.yaml; done
python extract_outputs.py --config configs/vanilla.yaml configs/gcsc.yaml configs/proser.yaml
python evaluate_osr.py
```

## Where each result comes from

| Task | Main tables | Figures |
|---|---|---|
| 1 | `task1/results/summary_tables.md`, `task1_results.json`, `cue_conflict_predictions.csv`, `cue_conflicts/metadata.json` | `task1/results/plots/` |
| 2 | `task2/results/comparison_table.md`, `final_report.json`, `per_class_target.csv`, `controlled_study_dan.json`, `*_history.json` | `task2/results/plots/` |
| 3 | `task3/results/comparison_table.md`, `final_report.json` (incl. `erm_matches_task2`), `per_class_sketch.csv`, `controlled_study_dan_dg.json` | `task3/results/plots/` |
| 4 | `task4/results/table1_posthoc_scores_vanilla.md`, `table2_model_comparison.md`, `summary.md`, `per_class_acceptance.csv`, `failures_vanilla_mls.csv`, `thresholds.json` | `task4/results/*.png` |

**Protocol checks**
- No target labels are used for Task 2 model selection: checkpoints are chosen by mean source-val macro-F1, and `evaluate_final.py` is the first script that reads Sketch labels.
- No Sketch data is used for Task 3 training or selection. `train.py` loads only the source part of the split. `evaluate_sketch.py` verifies the split and checkpoint fingerprints and that ERM reproduces Task 2's Source-only numbers.
- No CIFAR-100 image influences Task 4 training or thresholds. Thresholds are set on CIFAR-10 validation data only, and `task4/results/frozen_manifest.json` records the checkpoint hashes frozen before unknowns were scored.

## Attribution

- **AdaIN** (Task 1) network definitions and pretrained weights: naoto0804/pytorch-AdaIN (MIT), an implementation of Huang & Belongie (2017).
- **OpenCLIP**: mlfoundations/open_clip. torchvision models and dataset loaders.
- **PACS copy**: MachineLearning2020/Homework3-PACS (original PACS by Li et al., 2017).
- **DANN / GRL** after Ganin et al. (2016); **CDAN** after Long et al. (2018); **MMD** after Long et al. (2015).
- The optional discriminator gradient penalty follows DomainBed (facebookresearch/DomainBed).
- **SAM** after Foret et al. (2021).
- **PROSER** after Zhou et al. (2021). Its detection score follows the authors' reference code (zhoudw-zdw/CVPR21-Proser).
- **RPL** (optional, not run) is adapted from iCGY96/ARPL.
- Parts of the code were written with the help of an LLM coding assistant, as permitted by the course policy. All experiments, results and the report are my own.