from typing import List
import os

import typer
import torch

import models
import unlearn
from loader.face_dataset import get_dataset


app = typer.Typer()


def prepare_unlearn(model: str, weight: str, data_dir: str, splits: List[str], device: str, batch_size: int):
    workdir = os.path.join("./output", data_dir.split("/")[-2])
    # save_path = os.path.join(workdir, f"finetune_best.pt")
    save_path = workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    model = getattr(models, model)
    model = model()
    model.load_state_dict(torch.load(weight))
    model = model.to(device)

    retain_loader, forget_loader, valid_loader = get_dataset(
        batch_size=batch_size, data_dir=data_dir, splits=splits
    )
    return model, retain_loader, forget_loader, valid_loader, save_path


@app.command()
def finetune(model: str, weight: str,
             data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
             device: str = "cpu", batch_size: int = 64,
             epochs: int = 5, lr: float = 1e-5):
    model, retain_loader, forget_loader, valid_loader, save_path = prepare_unlearn(
        model, weight, data_dir, splits, device, batch_size)

    save_path = os.path.join(save_path, "finetune.pt")
    unlearn.finetune(model, retain_loader,
                     forget_loader, valid_loader,
                     device, save_path, epochs, lr)


@app.command()
def relabel(model: str, weight: str,
            data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
            device: str = "cpu", batch_size: int = 64,
            epochs: int = 5, lr: float = 1e-5):
    model, retain_loader, forget_loader, valid_loader, save_path = prepare_unlearn(
        model, weight, data_dir, splits, device, batch_size)

    save_path = os.path.join(save_path, "relabel.pt")
    unlearn.relabel(model, retain_loader,
                    forget_loader, valid_loader,
                    device, save_path, epochs, lr)


@app.command()
def hello(name: str):
    print(f"Hello {name}")


if __name__ == "__main__":
    app()
