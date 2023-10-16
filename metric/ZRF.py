from torch.nn import functional as F
import torch


def JSDiv(p, q):
    m = (p + q) / 2
    return 0.5 * F.kl_div(torch.log(p), m, reduction='batchmean') + 0.5 * F.kl_div(torch.log(q), m, reduction='batchmean')


# ZRF/UnLearningScore https://arxiv.org/abs/2205.08096
def ZRF(tmodel, gold_model, forget_dl, device):
    model_preds = []
    gold_model_preds = []
    tmodel = tmodel.to(device)
    gold_model = gold_model.to(device)
    with torch.no_grad():
        for batch in forget_dl:
            X, y = batch['image'], batch['age']
            X, y = X.to(device), y.to(device)
            model_output = tmodel(X)
            gold_model_output = gold_model(X)
            model_preds.append(F.softmax(model_output, dim=1).detach().cpu())
            gold_model_preds.append(
                F.softmax(gold_model_output, dim=1).detach().cpu())

    model_preds = torch.cat(model_preds, axis=0)
    gold_model_preds = torch.cat(gold_model_preds, axis=0)
    return 1 - JSDiv(model_preds, gold_model_preds)
