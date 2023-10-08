import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        # First convolutional filter. Batch norm and Relu activation are applied after.
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional filter. Batch norm and Relu activation are applied after.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # F(x) + x
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=Block, layers=[2, 2, 2, 2], num_classes=11):
        super().__init__()

        self.inplanes = 64

        # Changed kernel: 7->5; stride: 2->1, padding: 3->2
        # This is an adjustment to take into account the fact that our images
        # are 64x64, not 224x224.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=1, padding=2,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # Adaptive maxpool allows us to define the output dimensions.
        self.maxpool = nn.AdaptiveMaxPool2d((32, 32))

        # These are the layers that make up the bulk of ResNet. The first layer
        # does not halve the image dimensions (since stride=1 for the first layer
        # and 2 everywhere else).
        self.layer1 = self._make_layer(block, 64, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 1024, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):           # Notation: num_channelsxWxH
        x = self.conv1(x)           # In: 3x64x64     Out: 64x64x64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # In: 64x64x64    Out: 64x32x32

        x = self.layer1(x)          # In: 64x32x32    Out: 128x32x32
        x = self.layer2(x)          # In: 128x32x32   Out: 256x16x16
        x = self.layer3(x)          # In: 256x16x16   Out: 512x8x8
        x = self.layer4(x)          # In: 512x8x8     Out: 1024x4x4

        x = self.avgpool(x)         # In: 1024x4x4    Out: 1024x1x1
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Make the layer
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None

        # If the dimensions will be changed (either by having strides>1
        # or by increasing the number of channels), then make sure that the
        # identity mapping in adjusted accordingly)
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        # Create a list to store all the blocks
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)
