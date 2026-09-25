"""
CDAN: identical to DANN except the discriminator is conditioned on the
classifier's probability vector via g(x) = vec(f (x) p). Per the assignment:
no entropy conditioning, and neither f nor p is detached -- gradients from
the domain loss flow into both the backbone (via f) and the classifier head
(via p, including on TARGET examples, since p_t = softmax(head(feat_t)) is
computed with the current head weights and never detached).
"""
import torch
import torch.nn.functional as F

from methods.base import AdaptationMethod
from models.domain_discriminator import GradientReversalLayer, grl_alpha_schedule, cdan_feature


class CDANMethod(AdaptationMethod):
    def __init__(self, backbone, head, device, discriminator, max_alpha: float = 1.0):
        super().__init__(backbone, head, device)
        self.discriminator = discriminator
        self.grl = GradientReversalLayer()
        self.max_alpha = max_alpha

    def extra_parameters(self):
        return list(self.discriminator.parameters())

    def compute_step(self, source_batches: dict, target_batch, progress: float) -> dict:
        cls_loss, feat_s, logits_s, y_s = self._source_classification_loss(source_batches)
        p_s = F.softmax(logits_s, dim=-1)  # not detached

        x_t, _y_t_UNUSED = target_batch
        x_t = x_t.to(self.device)
        feat_t = self.backbone(x_t)
        logits_t = self.head(feat_t)
        p_t = F.softmax(logits_t, dim=-1)  # not detached

        g_s = cdan_feature(feat_s, p_s)
        g_t = cdan_feature(feat_t, p_t)

        alpha = grl_alpha_schedule(progress, self.max_alpha)
        g_all = torch.cat([g_s, g_t], dim=0)
        reversed_g = self.grl(g_all, alpha)
        domain_logits = self.discriminator(reversed_g)

        domain_labels = torch.cat([
            torch.zeros(feat_s.shape[0], dtype=torch.long, device=self.device),
            torch.ones(feat_t.shape[0], dtype=torch.long, device=self.device),
        ])
        domain_loss = F.cross_entropy(domain_logits, domain_labels)

        loss = cls_loss + domain_loss
        return {
            "loss": loss, "cls_loss": cls_loss.item(),
            "domain_loss": domain_loss.item(), "alpha": alpha,
        }
