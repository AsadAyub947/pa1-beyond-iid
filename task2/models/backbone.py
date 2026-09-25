"""
ResNet-18 backbone producing the 512-dim pooled feature, with the assignment's
required BatchNorm policy: running mean/variance frozen at their pretrained
ImageNet values for every method; gamma/beta remain trainable.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

FEAT_DIM = 512


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Everything up to (but excluding) the original 1000-way fc layer.
        self.body = nn.Sequential(*list(net.children())[:-1])

    def forward(self, x):
        feat = self.body(x)
        return feat.flatten(1)  # (B, 512)


def freeze_batchnorm_running_stats(model: nn.Module):
    """Call this every time AFTER model.train() (never instead of it).

    Puts every BatchNorm module into eval() mode -- so it uses (and does not
    update) its running_mean/running_var -- while leaving every other module
    in train() mode, and WITHOUT touching requires_grad on BatchNorm's
    weight/bias (gamma/beta), which therefore remain trainable via backprop.
    Do not call model.eval() on the whole model instead of this function --
    that would also disable dropout, which the domain discriminators need
    active during training.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
