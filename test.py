from typing import List
import torch
import typer

from loader.face_dataset import get_dataset
import models
import metric


def main(model: str, ckpt: str, data_dir: str = "./",
         splits: List[str] = ["data/train",
                              "data/validation_0/forget", "data/test"],
         device: str = "cpu", batch_size: int = 64):
    retain_loader, forget_loader, test_loader = get_dataset(
        batch_size, data_dir=data_dir, splits=splits
    )

    model = getattr(models, model)
    model = model()
    model.load_state_dict(torch.load(ckpt))
    model = model.to(device)

    mia_score = metric.MIA(retain_loader, forget_loader,
                           test_loader, model, device)
    print(f"MIA score of this pipeline: {mia_score}")


if __name__ == "__main__":
    typer.run(main)
