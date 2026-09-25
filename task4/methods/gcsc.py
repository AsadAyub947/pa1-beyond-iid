"""GCSC – 'good closed-set classifier' + MLS (Vaze et al., 2022).

Identical to the Vanilla recipe (initialisation, optimiser, schedule, batch
size, epochs, seed, checkpoint rule) with exactly one change: RandAugment
(num_ops=2, magnitude=9) inserted after crop+flip and before
ToTensor/Normalize. Evaluated with MLS on its 10 logits.
"""
from __future__ import annotations

from data.cifar10 import build_transform
from methods.vanilla import VanillaMethod


class GCSCMethod(VanillaMethod):
    name = "gcsc"

    def train_transform(self):
        ra = self.cfg.get("data", {}).get("randaugment") or {"num_ops": 2, "magnitude": 9}
        return build_transform(train=True, randaugment=ra)
