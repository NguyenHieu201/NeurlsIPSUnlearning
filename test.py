from loader.face_dataset import get_dataset
import models


retain_loader, forget_loader, valid_loader = get_dataset(
    64, data_dir="./data", splits=["forget_0", "forget_1", "forget_2"]
)
model = models.CNN()

batch_data = next(iter(retain_loader))
X, y = batch_data['image'], batch_data['age']
output = model(X)
print(output)
