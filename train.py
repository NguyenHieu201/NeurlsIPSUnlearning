import os

import typer
import torch

import models
from loader.face_dataset import get_dataset
from tqdm import tqdm


def train_step(model, batch_data, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()
    X, y = batch_data['image'], batch_data['age']
    X, y = X.to(device), y.to(device)
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validation_step(model, validation_loader, criterion, device):
    model.eval()
    losses = []
    for batch_data in validation_loader:
        X, y = batch_data['image'], batch_data['age']
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        loss = criterion(outputs, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main(model: str, data_dir: str, splits: [str, str], epochs: int = 10, device: str = "cpu", batch_size: int = 64, workdir: str = "./output"):
    model = getattr(models, model)
    model = model()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss()

    retain_loader, forget_loader, valid_loader = get_dataset(
        batch_size=batch_size, data_dir=data_dir, splits=splits
    )

    model = model.to(device)
    model.train()
    pbar = tqdm(range(epochs))
    best_loss = 1
    for epoch in pbar:
        for batch_data in retain_loader:
            train_loss = train_step(model, batch_data, criterion,
                                    optimizer, scheduler, device)
            pbar.set_postfix({"loss": train_loss})
        scheduler.step()

        # validation loop
        validation_loss = validation_step(
            model, valid_loader, criterion, device)
        if validation_loss < best_loss:
            best_loss = validation_loss
            pbar.set_description(f"best validation loss: {best_loss: .4f}")
            torch.save(model.state_dict(), os.path.join(workdir, "best.pt"))


if __name__ == "__main__":
    typer.run(main)
