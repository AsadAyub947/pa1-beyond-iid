"""Vanilla closed-set baseline: 10-way ResNet-18 trained with cross-entropy.

Every method in ``methods/`` implements the same small interface used by
``train.py`` and ``extract_outputs.py``:

    train_transform()            -> torchvision transform for the training split
    build_model()                -> nn.Module (randomly initialised or loaded)
    training_loss(model, x, y)   -> (loss, stats)  stats: correct / n for train acc
    known_logits(out)            -> the 10 known-class logits used for CSA / MLS
    extra_outputs(out)           -> extra arrays to cache (e.g. dummy logits)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from data.cifar10 import build_transform
from models.resnet_cifar import build_resnet18_cifar


class VanillaMethod:
    name = "vanilla"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.num_classes = int(cfg.get("model", {}).get("num_classes", 10))

    # -- data ------------------------------------------------------------- #
    def train_transform(self):
        return build_transform(train=True, randaugment=None)

    # -- model ------------------------------------------------------------ #
    def build_model(self) -> torch.nn.Module:
        return build_resnet18_cifar(num_classes=self.num_classes)

    def build_model_skeleton(self) -> torch.nn.Module:
        """Architecture only (no checkpoint loading) – used to restore saved weights."""
        return build_resnet18_cifar(num_classes=self.num_classes)

    # -- optimisation ----------------------------------------------------- #
    def training_loss(self, model, x, y):
        out = model(x)
        logits = out["logits"].float()
        loss = F.cross_entropy(logits, y)
        correct = (logits.argmax(1) == y).sum().item()
        return loss, {"correct": correct, "n": y.numel(), "loss_ce": loss.item()}

    # -- outputs ---------------------------------------------------------- #
    @staticmethod
    def known_logits(out: dict) -> torch.Tensor:
        return out["logits"]

    @staticmethod
    def extra_outputs(out: dict) -> dict:
        return {}
