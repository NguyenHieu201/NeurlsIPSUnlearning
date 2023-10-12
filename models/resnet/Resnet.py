import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = resnet18(weights=None, num_classes=5)

    def forward(self, inputs: torch.Tensor):
        return self.net(inputs)
