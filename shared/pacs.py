"""
PACS dataset loader.

PACS is not distributed via torchvision, so this expects the dataset to
already be present on disk in the standard per-domain/per-class ImageFolder
layout used by essentially every public PACS release:

    <root>/
        photo/
            dog/*.jpg
            elephant/*.jpg
            ...
        art_painting/
            dog/*.jpg
            ...
        cartoon/...
        sketch/...

Download it with shared/download_pacs.py (same copy for Task 2 and Task 3).
"""
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def list_domain_files(root: str, domain: str):
    """Return (filepaths, labels) for every image in a PACS domain folder,
    in a stable (sorted) order so index-based splits are reproducible."""
    domain_dir = os.path.join(root, domain)
    if not os.path.isdir(domain_dir):
        raise FileNotFoundError(
            f"PACS domain folder not found: {domain_dir}. "
            f"See task2/README.md for expected layout and download instructions.")
    filepaths, labels = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(domain_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                filepaths.append(os.path.join(cls_dir, fname))
                labels.append(CLASS_TO_IDX[cls])
    return filepaths, labels


class PACSDomainDataset(Dataset):
    """A single PACS domain, indexed by a fixed list of (global) sample
    indices so it can represent a train/val/target subset of that domain."""

    def __init__(self, filepaths, labels, indices, transform):
        self.filepaths = filepaths
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        real_idx = self.indices[i]
        img = Image.open(self.filepaths[real_idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[real_idx]
        return img, label
