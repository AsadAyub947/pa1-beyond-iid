"""
Sharpness-Aware Minimization (Foret et al., 2021), applied to plain ERM (no
domain-alignment term). Two forward/backward passes per batch:
  1. Compute the ERM loss/gradient at theta, take a normalized ascent step of
     radius rho to get theta + epsilon.
  2. Recompute the ERM loss/gradient AT THE PERTURBED POINT theta + epsilon,
     restore theta, then apply the base optimizer's update using the
     perturbed-point gradient.

BatchNorm running stats must stay frozen (via freeze_batchnorm_running_stats,
called once per epoch by train.py) across BOTH passes -- since nothing here
toggles train()/eval() mode between them, that's automatically satisfied.
"""
import torch

from methods.erm import erm_classification_loss


class SAMOptimizer:
    """Wraps a base optimizer (AdamW) with SAM's ascent-step / restore-and-
    update procedure. Not a torch.optim.Optimizer subclass -- deliberately a
    thin explicit wrapper so the two-pass control flow in SAMMethod.step is
    easy to follow instead of hidden behind a closure/opt.step(closure=...)."""

    def __init__(self, params, base_optimizer_cls=torch.optim.AdamW, rho: float = 0.05, **kwargs):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self.base_optimizer = base_optimizer_cls(self.params, **kwargs)
        self._e_ws = None

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @torch.no_grad()
    def ascent_step(self):
        """theta -> theta + epsilon, where epsilon = rho * grad / ||grad||_2
        (global norm across all parameters, per the assignment's formula)."""
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(2) for p in self.params if p.grad is not None]), 2)
        self._e_ws = []
        for p in self.params:
            if p.grad is None:
                self._e_ws.append(None)
                continue
            e_w = p.grad * (self.rho / (grad_norm + 1e-12))
            p.add_(e_w)
            self._e_ws.append(e_w)

    @torch.no_grad()
    def restore_and_step(self):
        """Undo the epsilon perturbation (back to the original theta), then
        apply the base optimizer's update using the gradient that was just
        computed AT the perturbed point (this is the defining trick of SAM:
        the update direction comes from theta+epsilon, but is applied to
        theta)."""
        for p, e_w in zip(self.params, self._e_ws):
            if e_w is not None:
                p.sub_(e_w)
        self.base_optimizer.step()
        self._e_ws = None


class SAMMethod:
    def __init__(self, backbone, head, device, rho: float = 0.05):
        self.backbone = backbone
        self.head = head
        self.device = device
        self.rho = rho  # controlled study sweeps this over {0.01, 0.05, 0.1}

    def extra_parameters(self):
        return []

    def make_optimizer(self, params, lr, weight_decay):
        return SAMOptimizer(params, base_optimizer_cls=torch.optim.AdamW,
                             rho=self.rho, lr=lr, weight_decay=weight_decay)

    def step(self, optimizer: SAMOptimizer, source_batches: dict) -> dict:
        # ---- First pass: gradient at theta, then ascend to theta + epsilon ----
        optimizer.zero_grad()
        loss1, _, _, _ = erm_classification_loss(self.backbone, self.head, source_batches, self.device)
        loss1.backward()
        optimizer.ascent_step()

        # ---- Second pass: gradient AT theta+epsilon (fresh forward pass --
        # parameters have changed, so activations MUST be recomputed) ----
        optimizer.zero_grad()
        loss2, _, _, _ = erm_classification_loss(self.backbone, self.head, source_batches, self.device)
        loss2.backward()
        optimizer.restore_and_step()

        return {"loss": loss1.item(), "loss_at_perturbed_point": loss2.item()}
