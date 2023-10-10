import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

from .parameter_pertuber import ParameterPerturber


def unlearning(
        net,
        retain_loader,
        forget_loader,
        val_loader, device: str = "cpu", save_path: str = "./output/ssd.pt"):

    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    alpha = 1  # alpha in the paper
    lambda_ = 10  # lambda in the paper
    selection_weighting = 10 * alpha

    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": lambda_,
        "selection_weighting": selection_weighting,
    }

    full_train_dl = DataLoader(
        ConcatDataset((retain_loader.dataset, forget_loader.dataset)),
        batch_size=64,
    )

    pdr = ParameterPerturber(net, optimizer, device, parameters)

    net = net.eval()

    sample_importances = pdr.calc_importance(forget_loader)
    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)

    net.eval()
    torch.save(net.state_dict(), save_path)
