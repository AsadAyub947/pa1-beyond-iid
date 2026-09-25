"""
Frozen-backbone feature extractors + linear classifier heads + CLIP zero-shot.

All three backbones expose a common interface:
    features = backbone.extract_features(img01_batch)   # (B, feat_dim)
where img01_batch is a (B,3,224,224) tensor in [0,1] (common representation);
each backbone internally applies its OWN required normalization.
"""
import copy

import open_clip
import torch
import torch.nn as nn
from torchvision.models import (resnet50, ResNet50_Weights,
                                 vit_b_16, ViT_B_16_Weights)

from data.transforms import normalize_for_model


class FrozenResNet50(nn.Module):
    name = "resnet50"
    feat_dim = 2048

    def __init__(self, device):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2
        net = resnet50(weights=weights)
        # Global-average-pooled feature = everything up to (and including) avgpool.
        self.body = nn.Sequential(*list(net.children())[:-1]).to(device).eval()
        for p in self.body.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def extract_features(self, img01_batch: torch.Tensor) -> torch.Tensor:
        x = normalize_for_model(img01_batch.to(self.device), self.name)
        feat = self.body(x)
        return feat.flatten(1)


class FrozenViTB16(nn.Module):
    name = "vit_b_16"
    feat_dim = 768

    def __init__(self, device):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.net = vit_b_16(weights=weights).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def extract_features(self, img01_batch: torch.Tensor) -> torch.Tensor:
        x = normalize_for_model(img01_batch.to(self.device), self.name)
        # Replicate torchvision's internal ViT forward up to (but excluding)
        # the classification head, returning the final class token.
        x = self.net._process_input(x)
        n = x.shape[0]
        batch_class_token = self.net.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.net.encoder(x)
        cls_token = x[:, 0]
        return cls_token


class FrozenCLIPViTB32(nn.Module):
    name = "clip_vit_b32"
    feat_dim = 512

    def __init__(self, device, pretrained="openai"):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=pretrained)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.device = device

    @torch.no_grad()
    def extract_features(self, img01_batch: torch.Tensor) -> torch.Tensor:
        x = normalize_for_model(img01_batch.to(self.device), self.name)
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def zero_shot_predict(self, img01_batch: torch.Tensor, class_names,
                           prompt_template="a photo of a {class}."):
        prompts = [prompt_template.format(**{"class": c}) for c in class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        text_feat = self.model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        img_feat = self.extract_features(img01_batch)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feat @ text_feat.t()
        probs = logits.softmax(dim=-1)
        conf, pred = probs.max(dim=-1)
        return pred.cpu(), conf.cpu(), probs.cpu()


def build_backbone(name: str, device):
    if name == "resnet50":
        return FrozenResNet50(device)
    if name == "vit_b_16":
        return FrozenViTB16(device)
    if name == "clip_vit_b32":
        return FrozenCLIPViTB32(device)
    raise ValueError(f"Unknown backbone: {name}")


class LinearHead(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_head(train_feats, train_labels, val_feats, val_labels,
                       feat_dim, num_classes, device, seed=6304,
                       lr=1e-3, weight_decay=1e-4, max_epochs=50,
                       patience=5, batch_size=64):
    """Train a single linear classifier on precomputed, frozen features."""
    torch.manual_seed(seed)

    head = LinearHead(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_feats = train_feats.to(device)
    train_labels = train_labels.to(device)
    val_feats = val_feats.to(device)
    val_labels = val_labels.to(device)

    n = train_feats.shape[0]
    best_val_acc = -1.0
    best_state = None
    epochs_without_improve = 0

    g = torch.Generator(device="cpu").manual_seed(seed)

    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = train_feats[idx], train_labels[idx]
            optimizer.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats)
            val_pred = val_logits.argmax(dim=-1)
            val_acc = (val_pred == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"  [early stop] epoch {epoch}, best val acc = {best_val_acc:.4f}")
                break

    head.load_state_dict(best_state)
    head.eval()
    return head, best_val_acc
