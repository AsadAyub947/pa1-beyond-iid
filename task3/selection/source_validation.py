"""
Checkpoint selection using ONLY the three source validation domains --
mean macro-F1, exactly as in Task 2. Kept in its own module (rather than
inlined in train.py) specifically so that "what decides which checkpoint we
keep" is easy to audit and provably never imports or touches Sketch, per the
assignment's "keep source-only model selection separate from the script that
loads Sketch" structuring advice.
"""
import numpy as np
import torch

from evaluation.domain_metrics import compute_domain_metrics, mean_across_domains


@torch.no_grad()
def evaluate_on_source_val(backbone, head, val_loaders: dict, device) -> dict:
    backbone.eval()
    head.eval()
    per_domain = {}
    for domain, loader in val_loaders.items():
        all_preds, all_labels = [], []
        for x, y in loader:
            x = x.to(device)
            logits = head(backbone(x))
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(y.numpy())
        per_domain[domain] = compute_domain_metrics(
            np.concatenate(all_labels), np.concatenate(all_preds))
    return {"per_domain": per_domain, "mean": mean_across_domains(per_domain)}


class EarlyStopTracker:
    """Tracks best-so-far mean source-validation macro-F1 and whether to stop,
    per the shared 5-epochs-without-improvement rule."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_mean_f1 = -1.0
        self.epochs_without_improve = 0
        self.best_state = None

    def update(self, mean_f1: float, state_dict_fn) -> bool:
        """state_dict_fn: zero-arg callable returning the state dict to keep
        if this is a new best (called lazily so we don't clone tensors on
        every epoch, only on improvement). Returns True if training should
        stop now."""
        if mean_f1 > self.best_mean_f1:
            self.best_mean_f1 = mean_f1
            self.epochs_without_improve = 0
            self.best_state = state_dict_fn()
            return False
        self.epochs_without_improve += 1
        return self.epochs_without_improve >= self.patience
