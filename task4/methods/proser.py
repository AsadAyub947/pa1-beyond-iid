"""PROSER – Learning Placeholders for Open-Set Recognition (Zhou et al., CVPR 2021).

Initialisation: the selected Vanilla checkpoint + C = 5 randomly initialised
dummy classifiers (``model.dummy_fc``). The whole network is fine-tuned.

Following the paper, the dummy "class" K+1 is the strongest dummy response:
    f^(x) = [ z_1(x), ..., z_K(x), max_c d_c(x) ]        (K+1 outputs)

Each mini-batch is split into two equal halves.

First half – classifier placeholders (paper Eq. 2, beta = 1):
    l_1 = CE(f^(x), y) + beta * CE(f^(x) \\ y, K+1)
    where f^(x) \\ y masks the ground-truth logit (set to -1e9) so that, once the
    true class is excluded, the dummy slot must be the strongest response.

Second half – data placeholders (paper Eq. 4, gamma = 0.1):
    h~ = lambda*phi_pre(x_i) + (1-lambda)*phi_pre(x_j),  y_i != y_j,  lambda ~ Beta(2,2)
    l_2 = CE(f^(phi_post(h~)), K+1)
    with phi_pre = stem..layer2 and phi_post = layer3..pool (mixup after layer2).

Total loss:  L = l_1 + gamma * l_2.
No CIFAR-100 image is involved; checkpoints are selected by CIFAR-10
validation accuracy computed from the 10 known logits only.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from methods.manifold_mixup import manifold_mixup
from methods.vanilla import VanillaMethod
from models.resnet_cifar import build_resnet18_cifar
from utils import resolve


def with_dummy_slot(out: dict) -> torch.Tensor:
    """[z_1..z_K, max_c d_c] in float32."""
    z = out["logits"].float()
    d = out["dummy_logits"].float().max(dim=1, keepdim=True).values
    return torch.cat([z, d], dim=1)


class ProserMethod(VanillaMethod):
    name = "proser"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        p = cfg.get("proser", {})
        self.num_dummy = int(p.get("num_dummy", 5))
        self.beta = float(p.get("beta", 1.0))
        self.gamma = float(p.get("gamma", 0.1))
        self.mix_alpha = float(p.get("mixup_alpha", 2.0))
        self.lambda_per_sample = bool(p.get("lambda_per_sample", False))
        self.init_checkpoint = p.get("init_checkpoint", "checkpoints/vanilla/best.pt")

    # -- model ------------------------------------------------------------ #
    def build_model(self) -> torch.nn.Module:
        path = resolve(self.init_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"PROSER needs the selected Vanilla checkpoint at {path}. "
                                    "Train Vanilla first: python train.py --config configs/vanilla.yaml")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_resnet18_cifar(num_classes=self.num_classes)
        model.load_state_dict(ckpt["model"])
        model.add_dummy_classifiers(self.num_dummy)  # randomly initialised (seeded)
        print(f"[proser] initialised from {path} (epoch {ckpt.get('epoch')}, "
              f"val_acc {ckpt.get('val_acc', float('nan')):.4f}) + {self.num_dummy} dummy classifiers")
        return model

    def build_model_skeleton(self) -> torch.nn.Module:
        return build_resnet18_cifar(num_classes=self.num_classes, num_dummy=self.num_dummy)

    # -- optimisation ----------------------------------------------------- #
    def training_loss(self, model, x, y):
        b = x.size(0)
        half = b // 2
        x_cp, y_cp = x[:half], y[:half]      # classifier placeholders
        x_dp, y_dp = x[half:], y[half:]      # data placeholders (manifold mixup)
        k = self.num_classes
        stats = {}

        # ---- classifier placeholders --------------------------------------
        out = model(x_cp)
        full = with_dummy_slot(out)
        loss_known = F.cross_entropy(full, y_cp)
        masked = full.scatter(1, y_cp.view(-1, 1), -1e9)
        dummy_target = torch.full_like(y_cp, k)
        loss_cp = F.cross_entropy(masked, dummy_target)

        # ---- data placeholders --------------------------------------------
        h = model.forward_pre(x_dp)
        h_mix, valid = manifold_mixup(h, y_dp, alpha=self.mix_alpha, per_sample=self.lambda_per_sample)
        if h_mix.size(0) > 0:
            full_mix = with_dummy_slot(model.forward_from_pre(h_mix))
            loss_dp = F.cross_entropy(full_mix, torch.full((h_mix.size(0),), k, device=x.device,
                                                           dtype=torch.long))
            stats["mix_to_dummy"] = (full_mix.argmax(1) == k).float().mean().item()
        else:
            loss_dp = full.new_zeros(())
            stats["mix_to_dummy"] = float("nan")

        loss = loss_known + self.beta * loss_cp + self.gamma * loss_dp
        known = out["logits"].float()
        stats.update({
            "correct": (known.argmax(1) == y_cp).sum().item(), "n": y_cp.numel(),
            "loss_known": loss_known.item(), "loss_cp": loss_cp.item(), "loss_dp": loss_dp.item(),
            "dummy_second": (masked.argmax(1) == k).float().mean().item(),
        })
        return loss, stats

    # -- outputs ---------------------------------------------------------- #
    @staticmethod
    def extra_outputs(out: dict) -> dict:
        return {"dummy_logits": out["dummy_logits"]}
