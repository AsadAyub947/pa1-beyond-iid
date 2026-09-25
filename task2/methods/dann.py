"""
DANN: a domain discriminator on the 512-dim feature, trained adversarially
against the backbone via a gradient-reversal layer (GRL).
"""
import torch
import torch.nn.functional as F

from methods.base import AdaptationMethod
from models.domain_discriminator import GradientReversalLayer, grl_alpha_schedule


class DANNMethod(AdaptationMethod):
    def __init__(self, backbone, head, device, discriminator, max_alpha: float = 1.0):
        super().__init__(backbone, head, device)
        self.discriminator = discriminator
        self.grl = GradientReversalLayer()
        self.max_alpha = max_alpha  # swept in the controlled design study: {0.25, 0.5, 1}

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    def compute_step(self, source_batches: dict, target_batch, progress: float) -> dict:
        # Only source examples contribute to the classification loss.
        cls_loss, feat_s, logits_s, y_s = self._source_classification_loss(source_batches)

        x_t, _y_t_UNUSED = target_batch
        x_t = x_t.to(self.device)
        feat_t = self.backbone(x_t)

        # Both source and target contribute to the domain-classification loss.
        alpha = grl_alpha_schedule(progress, self.max_alpha)
        feat_all = torch.cat([feat_s, feat_t], dim=0)
        reversed_feat = self.grl(feat_all, alpha)
        domain_logits = self.discriminator(reversed_feat)

        domain_labels = torch.cat([
            torch.zeros(feat_s.shape[0], dtype=torch.long, device=self.device),   # source = 0
            torch.ones(feat_t.shape[0], dtype=torch.long, device=self.device),    # target = 1
        ])
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        loss = cls_loss + domain_loss  # unit weight, per the assignment
        return {
            "loss": loss, "cls_loss": cls_loss.item(),
            "domain_loss": domain_loss.item(), "alpha": alpha,
        }
