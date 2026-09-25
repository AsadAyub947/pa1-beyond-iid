# Task 1 summary tables

## Clean, colour and patch-shuffle (accuracy % / consistency %)

| Model | Clean acc | Clean F1 | Clean conf | Gray acc (Δ) | Gray cons | Hue acc (Δ) | Hue cons | Shuffle acc (Δ) | Shuffle cons |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 97.6 | 97.6 | 0.962 | 94.0 (-3.6) | 95.2 | 94.8 (-2.8) | 96.2 | 89.6 (-8.0) | 90.6 |
| ViT-B/16 | 97.6 | 97.6 | 0.973 | 94.8 (-2.8) | 97.2 | 93.6 (-4.0) | 95.2 | 90.8 (-6.8) | 92.4 |
| CLIP head | 97.4 | 97.4 | 0.654 | 93.6 (-3.8) | 94.0 | 92.8 (-4.6) | 94.0 | 83.4 (-14.0) | 84.0 |
| CLIP zero-shot | 96.4 | 96.4 | 0.936 | 91.8 (-4.6) | 92.2 | 92.8 (-3.6) | 93.4 | 81.4 (-15.0) | 82.6 |

## Cue conflict

| Model | N shape | N texture | N other | Shape bias % | Coverage % |
|---|---|---|---|---|---|
| ResNet-50 | 69 | 57 | 74 | 54.8 | 63.0 |
| ViT-B/16 | 104 | 27 | 69 | 79.4 | 65.5 |
| CLIP head | 121 | 19 | 60 | 86.4 | 70.0 |
| CLIP zero-shot | 124 | 21 | 55 | 85.5 | 72.5 |

Generation: backend=adain_decoder, style strength=1.0, style centre crop=0.7, accepted=200, rejected=2 {'R1_degenerate': 0, 'R2_no_texture_transfer': 2, 'R3_shape_lost': 0, 'R4_manual': 0}

Example groups available: {'all shape': 54, 'disagree': 55, 'texture wins': 76, 'all other': 15}

## Translation (mean over 4 directions)

| Model | 0px acc / cons | 8px acc / cons | 16px acc / cons | 32px acc / cons |
|---|---|---|---|---|
| ResNet-50 | 97.6 / 100.0 | 96.7 / 98.5 | 96.7 / 98.4 | 96.2 / 97.7 |
| ViT-B/16 | 97.6 / 100.0 | 97.3 / 98.8 | 97.4 / 99.5 | 96.8 / 98.8 |
| CLIP head | 97.4 / 100.0 | 96.9 / 97.8 | 96.5 / 97.0 | 95.8 / 97.0 |
| CLIP zero-shot | 96.4 / 100.0 | 96.0 / 96.5 | 95.9 / 96.5 | 94.3 / 96.3 |

## Representation stability I_T (cosine)

| Backbone | grayscale | hue_rotation | patch_shuffle | cue_conflict_vs_source_content | translation_8px | translation_16px | translation_32px | random-pair baseline | same-class baseline |
|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.798 | 0.749 | 0.692 | 0.293 | 0.967 | 0.951 | 0.931 | 0.222 | 0.454 |
| ViT-B/16 | 0.732 | 0.716 | 0.628 | 0.274 | 0.945 | 0.976 | 0.944 | 0.069 | 0.291 |
| CLIP head | 0.888 | 0.884 | 0.745 | 0.701 | 0.978 | 0.963 | 0.967 | 0.592 | 0.737 |

## Linear heads

| Backbone | best val acc | best epoch | epochs run |
|---|---|---|---|
| ResNet-50 | 97.40 | 7 | 13 |
| ViT-B/16 | 97.90 | 5 | 11 |
| CLIP head | 98.60 | 6 | 12 |

CLIP zero-shot vs trained CLIP head agreement: clean 96.2%, grayscale 94.6%, hue_rotation 94.4%, patch_shuffle 85.8%, cue_conflict 71.0%
