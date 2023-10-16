from typing import List
import os

import typer
import torch

import models
import unlearn
from loader.face_dataset import get_dataset


app = typer.Typer()


def prepare_unlearn(model: str, weight: str, data_dir: str, splits: List[str], device: str, batch_size: int, workdir: str):
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    model = getattr(models, model)
    model = model()
    model.load_state_dict(torch.load(weight))
    model = model.to(device)

    retain_loader, forget_loader, valid_loader = get_dataset(
        batch_size=batch_size, data_dir=data_dir, splits=splits
    )
    return model, retain_loader, forget_loader, valid_loader


@app.command()
def finetune(model: str, weight: str,
             data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
             device: str = "cpu", batch_size: int = 64,
             epochs: int = 5, lr: float = 1e-5, workdir: str = "./output", repeat: int = 1):
    model, retain_loader, forget_loader, valid_loader = prepare_unlearn(
        model, weight, data_dir, splits, device, batch_size, workdir)

    for i in range(repeat):
        torch.manual_seed(i)
        save_path = os.path.join(workdir, f"finetune_seed_{i}.pt")
        unlearn.finetune(model, retain_loader,
                         forget_loader, valid_loader,
                         device, save_path, epochs, lr)


@app.command()
def relabel(model: str, weight: str,
            data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
            device: str = "cpu", batch_size: int = 64,
            epochs: int = 5, lr: float = 1e-5, workdir: str = "./output", repeat: int = 1):
    model, retain_loader, forget_loader, valid_loader = prepare_unlearn(
        model, weight, data_dir, splits, device, batch_size, workdir)

    for i in range(repeat):
        torch.manual_seed(i)
        save_path = os.path.join(workdir, f"relabel_seed_{i}.pt")
        unlearn.relabel(model, retain_loader,
                        forget_loader, valid_loader,
                        device, save_path, epochs, lr)


@app.command()
def ssd(model: str, weight: str,
        data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
        device: str = "cpu", batch_size: int = 64,
        epochs: int = 5, lr: float = 1e-5, workdir: str = "./output", repeat: int = 1):
    model, retain_loader, forget_loader, valid_loader = prepare_unlearn(
        model, weight, data_dir, splits, device, batch_size, workdir)

    for i in range(repeat):
        torch.manual_seed(i)
        save_path = os.path.join(workdir, f"ssd_seed_{i}.pt")
        unlearn.ssd(model, retain_loader, forget_loader,
                    valid_loader, device, save_path)


@app.command()
def stn(model: str, baseline: str, data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
        device: str = "cpu", batch_size: int = 64, alpha: int = 0.1,
        epochs: int = 1, lr: float = 1e-5, workdir: str = "./output", repeat: int = 1):
    _, retain_loader, forget_loader, valid_loader = prepare_unlearn(
        model, baseline, data_dir, splits, device, batch_size, workdir
    )
    model = getattr(models, model)

    for i in range(repeat):
        torch.manual_seed(i)
        save_path = os.path.join(workdir, f"stn_seed_{i}.pt")
        baseline_net = model()
        baseline_net.load_state_dict(torch.load(baseline))

        unlearn_net = model()
        unlearn_net.load_state_dict(torch.load(baseline))

        init_net = model()
        unlearn.stn(baseline_net, unlearn_net, init_net,
                    retain_loader, forget_loader,
                    lr, epochs, alpha, device, save_path)


if __name__ == "__main__":
    app()
