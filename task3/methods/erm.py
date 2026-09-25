"""
ERM is NOT retrained in Task 3 -- the assignment requires reusing Task 2's
Source-only checkpoint unchanged. This file therefore provides two things
instead of a training method:
  1. `erm_classification_loss`, the plain domain-balanced cross-entropy loss,
     reused as the base term inside DAN-DG's loss (L_DAN-DG = L_ERM + ...).
  2. `load_erm_checkpoint`, which loads Task 2's saved source_only.pt into
     Task 3's (structurally identical) backbone/head classes.
"""
import torch
import torch.nn.functional as F

from models.backbone import ResNet18Backbone, FEAT_DIM
from models.classifier_head import ClassifierHead, NUM_CLASSES


def erm_classification_loss(backbone, head, source_batches: dict, device):
    """source_batches: {domain_name: (x, y)}. Pools all source domains'
    labeled examples into one cross-entropy loss (domain-balanced because the
    caller constructs equal per-domain batches, per L_ERM = (1/3) sum_e R_e --
    with equal per-domain batch sizes, plain pooled CE already weights every
    domain equally in expectation)."""
    xs, ys, feats_by_domain = [], [], {}
    for domain, (x, y) in source_batches.items():
        x = x.to(device)
        feat = backbone(x)
        feats_by_domain[domain] = feat
        xs.append(feat)
        ys.append(y.to(device))
    all_feat = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    logits = head(all_feat)
    loss = F.cross_entropy(logits, all_y)
    return loss, feats_by_domain, logits, all_y


def load_erm_checkpoint(task2_source_only_ckpt_path: str, device):
    """Loads Task 2's Source-only checkpoint verbatim -- this IS the Task 3
    ERM baseline; it must not be retrained under Task 3's settings."""
    ckpt = torch.load(task2_source_only_ckpt_path, map_location=device)
    backbone = ResNet18Backbone().to(device)
    head = ClassifierHead(FEAT_DIM, NUM_CLASSES).to(device)
    backbone.load_state_dict(ckpt["state"]["backbone"])
    head.load_state_dict(ckpt["state"]["head"])
    return backbone, head
