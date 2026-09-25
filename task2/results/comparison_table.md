Accuracies / F1 / separability in %.

| method | photo_val_acc | art_painting_val_acc | cartoon_val_acc | mean_source_acc | mean_source_f1 | target_acc | target_f1 | target_acc_change_vs_source_only | domain_separability | diverged |
|---|---|---|---|---|---|---|---|---|---|---|
| source_only | 96.40 | 94.15 | 94.24 | 94.93 | 94.89 | 57.29 | 58.40 | 0.00 | 99.73 | False |
| dan | 96.40 | 92.93 | 93.82 | 94.38 | 94.35 | 68.69 | 63.31 | 11.40 | 86.95 | False |
| dann | 45.05 | 22.20 | 32.41 | 33.22 | 25.66 | 30.92 | 20.78 | -26.37 | 97.53 | True |
| cdan | 43.54 | 26.34 | 22.60 | 30.83 | 21.14 | 2.04 | 0.57 | -55.26 | 99.73 | True |
| dann_stabilized | 88.89 | 73.66 | 72.28 | 78.28 | 77.25 | 15.47 | 3.83 | -41.82 | 99.86 | True |
| cdan_stabilized | 96.70 | 90.00 | 91.26 | 92.65 | 92.85 | 43.17 | 48.56 | -14.13 | 98.63 | False |
