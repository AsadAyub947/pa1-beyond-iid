import torch.nn as nn

NUM_CLASSES = 7


class ClassifierHead(nn.Module):
    def __init__(self, feat_dim: int = 512, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, feat):
        return self.fc(feat)
