"""CIFAR-appropriate ResNet-18 shared by every method in Task 4.

Built from ``torchvision.models.resnet18`` (random initialisation) with the two
required changes:
  * the ImageNet 7x7 / stride-2 stem convolution is replaced by a 3x3 / stride-1
    convolution;
  * the initial max-pooling layer is removed (``nn.Identity``).
Inputs stay at their original 32x32 resolution.

The forward pass is split so manifold mixup can be applied after ``layer2``:
    forward_pre(x)   : stem -> layer1 -> layer2           (phi_pre)
    forward_post(h)  : layer3 -> layer4 -> avgpool -> f(x) (phi_post, 512-d)
    head(f)          : known-class logits z(x) (+ optional dummy logits)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision

FEATURE_DIM = 512


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, num_dummy: int = 0):
        super().__init__()
        base = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")
        base.maxpool = nn.Identity()

        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.fc = base.fc  # known-class classifier (may be replaced, e.g. RPL head)
        self.num_classes = num_classes
        self.dummy_fc: Optional[nn.Linear] = None
        if num_dummy > 0:
            self.add_dummy_classifiers(num_dummy)

    # ------------------------------------------------------------------ #
    def add_dummy_classifiers(self, num_dummy: int) -> None:
        """Append ``num_dummy`` randomly initialised dummy (placeholder) classifiers."""
        self.dummy_fc = nn.Linear(FEATURE_DIM, num_dummy)

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        return self.layer2(x)

    def forward_post(self, h: torch.Tensor) -> torch.Tensor:
        h = self.layer4(self.layer3(h))
        return torch.flatten(self.avgpool(h), 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_post(self.forward_pre(x))

    def head(self, f: torch.Tensor) -> dict:
        out = {"features": f, "logits": self.fc(f)}
        if self.dummy_fc is not None:
            out["dummy_logits"] = self.dummy_fc(f)
        return out

    def forward(self, x: torch.Tensor) -> dict:
        return self.head(self.features(x))

    def forward_from_pre(self, h: torch.Tensor) -> dict:
        return self.head(self.forward_post(h))


def build_resnet18_cifar(num_classes: int = 10, num_dummy: int = 0) -> ResNet18CIFAR:
    return ResNet18CIFAR(num_classes=num_classes, num_dummy=num_dummy)
