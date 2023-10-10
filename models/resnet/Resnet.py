import torch
import torch.nn as nn
from torchvision.models import resnet18


class Resnet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = resnet18(weights=None, num_classes=5)

    def forward(self, inputs: torch.Tensor):
        if inputs.max() > 1:
            inputs = inputs / 255
        return self.net(inputs)
