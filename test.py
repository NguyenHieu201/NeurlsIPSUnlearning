import torch

from loader.face_dataset import get_dataset
import models
import metric


retain_loader, forget_loader, test_loader = get_dataset(
    64, data_dir="./", splits=["data/train.csv", "data/validation_0/forget.csv", "data/test.csv"]
)
model = models.CNN()
model.load_state_dict(torch.load(""))

mia_score = metric.mia(model, test_loader, forget_loader)
print(f"{mia_score}")
