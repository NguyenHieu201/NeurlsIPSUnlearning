import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .reconstruction_loss import ReconstructionLoss

# Paper: https://arxiv.org/abs/2308.14322


def knowledge_erasure(unlearn_net: nn.Module, init_net: nn.Module,
                      forget_loader: DataLoader,
                      lr: float = 0.01, epochs: int = 1, device="cpu"):
    optimizer = optim.SGD(unlearn_net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.KLDivLoss(reduction="batchmean")

    unlearn_net = unlearn_net.to(device)
    init_net = init_net.to(device)

    for epoch in range(epochs):
        unlearn_net.train()
        for batch in forget_loader:
            optimizer.zero_grad()
            inputs, targets = batch["image"], batch["age"]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = unlearn_net(inputs)
            init_outputs = init_net(inputs)

            outputs, init_outputs = torch.softmax(
                outputs, dim=1), torch.softmax(init_outputs, dim=1)

            loss = criterion(outputs, init_outputs)
            loss.backward()
            optimizer.step()
    return unlearn_net


def model_reconstruction(baseline_net: nn.Module, unlearn_net: nn.Module,
                         retain_loader: DataLoader,
                         lr: float = 0.01, epochs: int = 1, alpha: float = 0.1, device="cpu"):
    optimizer = optim.SGD(unlearn_net.parameters(), lr=lr, momentum=0.9)
    criterion = ReconstructionLoss(alpha=alpha)

    baseline_net = baseline_net.to(device)
    unlearn_net = unlearn_net.to(device)

    for epoch in range(epochs):
        unlearn_net.train()
        for batch in retain_loader:
            optimizer.zero_grad()
            inputs, targets = batch["image"], batch["age"]
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = unlearn_net(inputs)
            baseline_outputs = baseline_net(inputs)

            outputs = torch.softmax(outputs, dim=1)
            baseline_outputs = torch.softmax(baseline_outputs, dim=1)

            loss = criterion(outputs, baseline_outputs, targets)
            loss.backward()
            optimizer.step()


def stn(baseline_net: nn.Module, unlearn_net: nn.Module, init_net: nn.Module,
        retain_loader: DataLoader, forget_loader: DataLoader,
        lr: float = 0.01, epochs: int = 1, alpha: float = 0.1, device="cpu",
        save_path: str = "./output/stn.pt"):
    knowledge_erasure(unlearn_net, init_net, forget_loader, lr, epochs, device)
    model_reconstruction(baseline_net, unlearn_net,
                         retain_loader, lr, epochs, alpha, device)
    unlearn_net.eval()
    torch.save(unlearn_net.state_dict(), save_path)
