"""
Generate shape/texture cue-conflict images using AdaIN (Huang & Belongie, 2017).

Implementation note (a documented design choice, as the assignment permits):
this uses an OPTIMIZATION-BASED variant of AdaIN rather than the original
feed-forward decoder network. A pretrained (frozen) VGG-19 encoder computes
content and style features; the content feature map is renormalized to the
style's channel-wise mean/std (the AdaIN operation itself, unchanged from the
original method); a pixel image is then optimized by gradient descent so that
its own VGG features match that AdaIN target. This avoids needing a
separately-trained decoder network while implementing the same core AdaIN
statistic-matching idea. It is slower per image than a feed-forward decoder
(configurable via `optim_steps`) but requires no extra pretrained weights
beyond torchvision's VGG-19.

If you have access to a pretrained AdaIN decoder checkpoint (e.g. from the
original repo), you can swap `optimize_to_match_adain_features` below for a
single decoder forward pass -- the AdaIN feature computation is unchanged.
"""
import argparse
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.datasets import STL10
from torchvision.models import vgg19, VGG19_Weights

from transforms import to_common_image, IMG_SIZE

VGG_LAYER_INDEX = {
    # torchvision vgg19().features indices for standard AdaIN layers
    "relu1_1": 1, "relu2_1": 6, "relu3_1": 11, "relu4_1": 20,
}


def build_vgg_encoder(layer_name: str, device):
    vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
    idx = VGG_LAYER_INDEX[layer_name]
    encoder = torch.nn.Sequential(*list(vgg.children())[: idx + 1]).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def vgg_preprocess(img01: torch.Tensor, device) -> torch.Tensor:
    return (img01.to(device) - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor, eps=1e-5,
          strength: float = 1.0) -> torch.Tensor:
    """Standard AdaIN: renormalize content feature stats to style feature stats,
    channel-wise. `strength` in [0,1] interpolates between content (0) and
    fully-restyled (1) feature statistics -- the assignment's 'style strength'
    experimental-design knob."""
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + eps
    normalized = (content_feat - c_mean) / c_std
    target = normalized * s_std + s_mean
    return strength * target + (1 - strength) * content_feat


def optimize_to_match_adain_features(encoder, content_img01, style_img01, device,
                                      steps=300, lr=0.05, strength=1.0):
    content_feat = encoder(vgg_preprocess(content_img01.unsqueeze(0), device))
    style_feat = encoder(vgg_preprocess(style_img01.unsqueeze(0), device))
    target_feat = adain(content_feat, style_feat, strength=strength).detach()

    # Initialize from the content image (keeps optimization well-behaved and
    # preserves shape, which is exactly what we want for a cue conflict).
    out = content_img01.clone().unsqueeze(0).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([out], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        out_clamped = out.clamp(0, 1)
        feat = encoder(vgg_preprocess(out_clamped, device))
        loss = F.mse_loss(feat, target_feat)
        loss.backward()
        optimizer.step()

    return out.detach().clamp(0, 1).squeeze(0).cpu()


def is_valid_stylization(img01: torch.Tensor, std_threshold: float) -> bool:
    """Visual rejection rule, fixed BEFORE model evaluation and independent of
    any model's predictions: reject degenerate outputs (near-uniform color,
    i.e. failed optimization / collapsed stylization) and any NaN/Inf pixels."""
    if torch.isnan(img01).any() or torch.isinf(img01).any():
        return False
    return img01.std().item() >= std_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--subset_indices", default="results/subset_indices.json")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.subset_indices) as f:
        subset = json.load(f)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cc_cfg = cfg["cue_conflict"]
    class_names = cfg["dataset"]["class_names"]
    n_classes = len(class_names)

    encoder = build_vgg_encoder(cc_cfg["adain_layer"], device)

    print("Loading STL-10 test partition to source content/style images "
          "(excluding the 500-image evaluation subset used for the other "
          "experiments, so cue-conflict source images don't duplicate those "
          "used elsewhere). Note: the official STL-10 train partition (5000 "
          "images) is fully consumed by the 80/20 split used to train the "
          "linear heads, so it is not a viable source pool here.")
    test_ds = STL10(root=cfg["dataset"]["root"], split="test", download=True)
    labels = np.array(test_ds.labels)
    eval_subset_idx = set(subset.get("test_subset_idx", []))
    by_class = {
        c: [i for i in np.where(labels == c)[0].tolist() if i not in eval_subset_idx]
        for c in range(n_classes)
    }

    all_pairs = list(itertools.combinations(range(n_classes), 2))
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    chosen_pairs = all_pairs[: cc_cfg["n_class_pairs"]]
    print(f"Chosen class pairs (unordered): "
          f"{[(class_names[a], class_names[b]) for a, b in chosen_pairs]}")

    target_total = cc_cfg["target_total_conflicts"]
    per_direction = max(1, target_total // (len(chosen_pairs) * 2))

    out_dir = os.path.join(cfg["paths"]["results_dir"], "cue_conflicts")
    os.makedirs(out_dir, exist_ok=True)

    accepted, rejected = 0, 0
    metadata = []
    img_counter = 0

    for (a, b) in chosen_pairs:
        for (content_cls, style_cls) in [(a, b), (b, a)]:
            content_pool = by_class[content_cls]
            style_pool = by_class[style_cls]
            n = min(per_direction, len(content_pool), len(style_pool))
            content_ids = rng.sample(content_pool, n)
            style_ids = rng.sample(style_pool, n)

            for c_id, s_id in zip(content_ids, style_ids):
                content_pil = Image.fromarray(test_ds.data[c_id].transpose(1, 2, 0))
                style_pil = Image.fromarray(test_ds.data[s_id].transpose(1, 2, 0))
                content01 = to_common_image(content_pil)
                style01 = to_common_image(style_pil)

                result = optimize_to_match_adain_features(
                    encoder, content01, style01, device,
                    steps=cc_cfg["optim_steps"], lr=cc_cfg["optim_lr"],
                    strength=cc_cfg["style_strength"],
                )

                if is_valid_stylization(result, cc_cfg["rejection_std_threshold"]):
                    fname = f"conflict_{img_counter:05d}.pt"
                    torch.save(result, os.path.join(out_dir, fname))
                    metadata.append({
                        "file": fname,
                        "content_label": int(content_cls),
                        "style_label": int(style_cls),
                        "content_class_name": class_names[content_cls],
                        "style_class_name": class_names[style_cls],
                        "source_content_idx": int(c_id),
                        "source_style_idx": int(s_id),
                    })
                    accepted += 1
                else:
                    rejected += 1
                img_counter += 1

    meta_path = os.path.join(out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "accepted": accepted,
            "rejected": rejected,
            "rejection_std_threshold": cc_cfg["rejection_std_threshold"],
            "class_pairs": chosen_pairs,
            "conflicts": metadata,
        }, f, indent=2)

    print(f"Accepted {accepted} valid cue-conflict images, rejected {rejected}.")
    print(f"Saved images + metadata under {out_dir}")
    if accepted < target_total:
        print(f"WARNING: accepted count ({accepted}) is below the target "
              f"({target_total}). Consider more class pairs, more optim_steps, "
              f"or a lower rejection_std_threshold.")


if __name__ == "__main__":
    main()
