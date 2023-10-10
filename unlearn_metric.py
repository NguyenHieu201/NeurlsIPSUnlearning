from typing import List
import torch
import typer

from loader.face_dataset import get_dataset
import models
import metric


def main(model: str, umodel: str, rmodel: str, data_dir: str = "./",
         splits: List[str] = ["data/train",
                              "data/validation_0/forget", "data/test"],
         device: str = "cpu", batch_size: int = 64):
    _, forget_loader, test_loader = get_dataset(
        batch_size, data_dir=data_dir, splits=splits
    )

    model = getattr(models, model)

    tmodel = model()
    tmodel.load_state_dict(torch.load(umodel))

    gmodel = model()
    gmodel.load_state_dict(torch.load(rmodel))

    score = metric.ZRF(tmodel, gmodel, forget_loader, device)
    print(f"Unlearning score of this pipeline: {score: .4f}")


if __name__ == "__main__":
    typer.run(main)
