"""
Identical architecture and BatchNorm policy to Task 2's models/backbone.py.
Duplicated here (rather than imported cross-task) so Task 3 is a
self-contained deliverable, per the suggested repo structure listing
task3/models/backbone.py as its own file. The ERM baseline, however, is NOT
retrained here -- see methods/erm.py -- its checkpoint is loaded directly
from Task 2's saved source_only.pt so both tasks reference the exact same
trained weights.
"""
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

FEAT_DIM = 512


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        feat = self.body(x)
        return feat.flatten(1)


def freeze_batchnorm_running_stats(model: nn.Module):
    """Same policy as Task 2: BatchNorm modules in eval() (frozen running
    stats), everything else stays in train(). Call this every time AFTER
    model.train(). Required during BOTH SAM forward/backward passes."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
