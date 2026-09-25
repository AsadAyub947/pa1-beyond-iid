### Table 2 – Trained-model comparison (MLS common score + PROSER placeholder score)

| Model | Score | CSA (%) | AUROC Near (%) | AUROC Far (%) | AUROC All (%) | τ (val 95th pct) | Test accept (%) | Near reject (%) | Far reject (%) | All reject (%) | FPR@95 Near (%) | FPR@95 Far (%) | FPR@95 All (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | MLS | 94.82 | 80.64 | 89.40 | 85.02 | -5.78 | 95.11 | 31.87 | 53.00 | 42.44 | 68.12 | 47.00 | 57.56 |
| GCSC | MLS | 94.96 | 81.44 | 89.77 | 85.60 | -5.989 | 95.32 | 32.50 | 55.50 | 44.00 | 67.50 | 44.50 | 56.00 |
| PROSER | MLS | 94.19 | 79.81 | 89.64 | 84.72 | -4.826 | 95.37 | 27.62 | 46.75 | 37.19 | 72.38 | 53.25 | 62.81 |
| PROSER | PROSER dummy (placeholder) | 94.19 | 79.33 | 88.51 | 83.92 | 0.6307 | 95.54 | 29.00 | 45.38 | 37.19 | 71.00 | 54.62 | 62.81 |
| PROSER | PROSER dummy margin (bias-calibrated) | 94.19 | 79.31 | 88.44 | 83.87 | 0.7621 | 95.44 | 29.12 | 46.00 | 37.56 | 70.88 | 54.00 | 62.44 |
