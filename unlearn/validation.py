import torch


@torch.no_grad()
def validation_step(model, validation_loader, device):
    model.eval()
    acc = 0
    cnt = 0
    for batch_data in validation_loader:
        X, y = batch_data['image'], batch_data['age']
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        acc += (outputs.argmax(axis=1) == y).sum().item()
        cnt += outputs.shape[0]
    return acc / cnt
