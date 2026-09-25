"""
Common preprocessing and controlled interventions.

Design principle (per the assignment): every intervention is constructed on a
common 224x224 RGB image in [0, 1], BEFORE any model-specific normalization is
applied. This guarantees every backbone receives pixel-identical clean and
transformed images (up to their own required normalization).
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF

IMG_SIZE = 224

# ImageNet normalization (used by torchvision ResNet-50 / ViT-B/16 IMAGENET1K weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# OpenAI CLIP normalization
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

_to_common = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),  # -> float tensor in [0, 1], shape (3, H, W)
])


def to_common_image(pil_img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to the common 224x224 RGB [0,1] tensor representation
    shared by ALL backbones and ALL interventions."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return _to_common(pil_img)


def normalize_for_model(img01: torch.Tensor, model_name: str) -> torch.Tensor:
    """Apply the model-specific normalization. img01 must be a (B,3,H,W) or
    (3,H,W) tensor in [0, 1]."""
    if model_name in ("resnet50", "vit_b_16"):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif model_name == "clip_vit_b32":
        mean, std = CLIP_MEAN, CLIP_STD
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    mean_t = torch.tensor(mean, device=img01.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=img01.device).view(-1, 1, 1)
    return (img01 - mean_t) / std_t


# ---------------------------------------------------------------------------
# Interventions (all operate on a (3,H,W) float tensor in [0,1] and return the
# same shape/range so they can be chained/composed and normalized afterward).
# ---------------------------------------------------------------------------

def grayscale_transform(img01: torch.Tensor) -> torch.Tensor:
    """Remove color; replicate to 3 channels so downstream shapes are unchanged."""
    gray = TF.rgb_to_grayscale(img01, num_output_channels=3)
    return gray


def hue_rotate_transform(img01: torch.Tensor, hue_shift: float = 0.25) -> torch.Tensor:
    """Rotate hue while preserving luminance and saturation structure
    (preserves object geometry; changes chromatic identity)."""
    return TF.adjust_hue(img01, hue_shift)


def translate_transform(img01: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Shift the image by (dx, dy) pixels using reflection padding followed by
    a shifted crop, so the output has the same size as the input and no
    border artifacts are introduced by zero-padding."""
    if dx == 0 and dy == 0:
        return img01
    c, h, w = img01.shape
    pad = max(abs(dx), abs(dy))
    padded = F.pad(img01.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
    top = pad - dy
    left = pad - dx
    shifted = padded[:, top:top + h, left:left + w]
    return shifted


def patch_shuffle_transform(img01: torch.Tensor, grid: int = 4, seed: int = 6304,
                             index: int = 0) -> torch.Tensor:
    """Shuffle the image's grid x grid patches using a per-image deterministic,
    non-identity permutation derived from (seed, index)."""
    c, h, w = img01.shape
    assert h % grid == 0 and w % grid == 0, "Image size must be divisible by grid."
    ph, pw = h // grid, w // grid
    n_patches = grid * grid

    rng = np.random.RandomState(seed + index)
    perm = rng.permutation(n_patches)
    # Ensure non-identity permutation (re-sample if identity, which is
    # astronomically unlikely for n_patches=16 but checked defensively).
    tries = 0
    while np.array_equal(perm, np.arange(n_patches)) and tries < 10:
        perm = rng.permutation(n_patches)
        tries += 1

    patches = img01.unfold(1, ph, ph).unfold(2, pw, pw)  # (c, grid, grid, ph, pw)
    patches = patches.contiguous().view(c, n_patches, ph, pw)
    shuffled = patches[:, perm, :, :]
    shuffled = shuffled.view(c, grid, grid, ph, pw)
    shuffled = shuffled.permute(0, 1, 3, 2, 4).contiguous().view(c, h, w)
    return shuffled
