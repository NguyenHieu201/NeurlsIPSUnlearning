import os
from typing import List

import typer
import torch

import models
import unlearn
from loader.face_dataset import get_dataset


def main(model: str, weight: str, alg: str,
         data_dir: str, splits: List[str] = ["forget", "retrain", "validation"],
         epochs: int = 10, device: str = "cpu", batch_size: int = 64):
    workdir = os.path.join("./output", data_dir.split("/")[-2])
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    model = getattr(models, model)
    model = model()
    model.load_state_dict(torch.load(weight))
    model = model.to(device)

    unlearn_algorithm = getattr(unlearn, alg)
    retain_loader, forget_loader, valid_loader = get_dataset(
        batch_size=batch_size, data_dir=data_dir, splits=splits
    )
    model = unlearn_algorithm(model, retain_loader,
                              forget_loader, valid_loader,
                              device)
    torch.save(model.state_dict(), os.path.join(workdir, f"{alg}_best.pt"))


if __name__ == "__main__":
    typer.run(main)
