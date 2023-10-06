import torch

from loader.face_dataset import get_dataset
import models
import metric
from unlearn.relabel.relabel import relabel


# retain_loader, forget_loader, test_loader = get_dataset(
#     64, data_dir="./", splits=["data/train", "data/validation_0/forget", "data/test"]
# )

# batch_data = iter(next(retain_loader))
# X, y = batch_data["image"], batch_data["age"]
# print(relabel(y))

y = torch.randint(0, 10, (64, 1))
print(relabel(y))

# model = models.CNN()
# model.load_state_dict(torch.load(""))

# mia_score = metric.mia(model, test_loader, forget_loader)
# print(f"{mia_score}")
