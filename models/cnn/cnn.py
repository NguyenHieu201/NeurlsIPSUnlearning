import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
        )

        self.last_layer = nn.Sequential(nn.Linear(131072, 1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 11)
                                        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = self.last_layer(x)
        return x
