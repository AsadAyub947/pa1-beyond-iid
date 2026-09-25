"""
Common interface every method (source_only, dan, dann, cdan) implements, so
that train.py contains exactly one training loop shared by all of them, and
this directory contains only method-specific loss logic (per the assignment's
suggested repo structure comment: "keep method-specific losses separate from
the common training loop").
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptationMethod(ABC):
    def __init__(self, backbone, head, device):
        self.backbone = backbone
        self.head = head
        self.device = device

    def extra_parameters(self):
        """Parameters beyond backbone+head (e.g. a domain discriminator).
        Returned separately so train.py can put everything in one optimizer."""
        return []

    @abstractmethod
    def compute_step(self, source_batches: dict, target_batch, progress: float) -> dict:
        """
        source_batches: {domain_name: (x, y)} -- ALL source labels usable.
        target_batch: (x, y) where y must NEVER be used for anything other
            than final post-hoc evaluation -- adaptation methods only ever
            use target_batch[0] (the images).
        progress: float in [0, 1], current training progress (for the GRL
            alpha schedule).

        Returns a dict with at least {"loss": <scalar tensor to backward>}
        plus any additional scalar floats for logging (e.g. "cls_loss",
        "domain_loss", "mmd").
        """
        raise NotImplementedError

    def _source_classification_loss(self, source_batches: dict):
        """Shared by every method: plain cross-entropy over ALL source
        domains' labeled examples, pooled together."""
        xs, ys = [], []
        for domain, (x, y) in source_batches.items():
            xs.append(x.to(self.device))
            ys.append(y.to(self.device))
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        feat = self.backbone(x)
        logits = self.head(feat)
        loss = F.cross_entropy(logits, y)
        return loss, feat, logits, y
