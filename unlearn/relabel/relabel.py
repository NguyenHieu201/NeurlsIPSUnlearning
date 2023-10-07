import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..validation import validation_step


@torch.no_grad()
def relabel(targets: torch.Tensor) -> torch.Tensor:
    max_age_group = int(targets.max().item())
    step_label = torch.randint(low=1, high=max_age_group, size=targets.size())
    relabel_targets = targets + step_label
    relabel_targets[relabel_targets >= max_age_group] -= max_age_group
    return relabel_targets


def unlearning(net, retain, forget, validation, device: str = "cpu", save_path: str = "./output/best.pt",
               epochs=5, lr=1e-3):
    """Unlearning by fine-tuning.

    Fine-tuning is a very simple algorithm that trains using only
    the retain set.

    Args:
      net : nn.Module.
        pre-trained model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    net.train()

    best_acc = 0
    pbar = tqdm(range(epochs))
    for _ in pbar:
        net.train()
        for batch_data in forget:
            inputs, targets = batch_data['image'], batch_data['age']
            inputs = inputs.to(device)

            targets = relabel(targets)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item(): .3f}"})
        scheduler.step()

        # validation step - keep the utility of the model
        validation_acc = validation_step(net, validation, device)
        if validation_acc > best_acc:
            best_acc = validation_acc
            pbar.set_description(f"Best acc: {validation_acc}")
            torch.save(net.state_dict(), save_path)
