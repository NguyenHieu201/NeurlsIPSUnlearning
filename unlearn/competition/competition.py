import os
from typing import List
from sklearn.linear_model import LogisticRegression

import numpy as np
import torch
import math

import models
from loader.face_dataset import get_loader


INF = 1e8
DELTA = 0.1


def get_results(model, loader, device: str = "cpu"):
    preds = []
    model = model.to(device)
    model.eval()
    acc = 0
    for batch in loader:
        X, y = batch["image"], batch["age"]
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        preds.append(outputs.detach().cpu().squeeze())
        acc += (outputs.argmax(axis=1) == y).sum().item()
    preds = torch.concat(preds)
    return preds, acc / len(loader.dataset)


def example_epsilon(clf: LogisticRegression, uf_pred, rf_pred) -> float:
    epsilon = 0
    # uf: 0, rf: 1
    mia_uf = clf.predict(uf_pred)
    mia_rf = clf.predict(rf_pred)

    total_pred = mia_uf.shape[0] + mia_rf.shape[0]
    fpr = (mia_uf == 1).sum() / total_pred
    fnr = (mia_rf == 0).sum() / total_pred

    if (fpr == 0) and (fnr == 0):
        epsilon = INF
    elif (fpr == 0) or (fnr == 0):
        epsilon = 0
    else:
        epsilon1 = np.log(1 - DELTA - fpr) - np.log(fnr)
        epsilon2 = np.log(1 - DELTA - fnr) - np.log(fpr)
        epsilon = epsilon1 + epsilon2
        epsilon = epsilon

    epsilon_bin = math.floor(epsilon / 0.5 - 1e-8)
    return 1 / math.pow(2, epsilon_bin)


def forget_quality(uv_preds: List[torch.Tensor], rv_preds: List[torch.Tensor], uf_preds, rf_preds) -> float:
    num_forget = uf_preds[0].shape[0]
    H = 0
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )

    uv_preds = torch.concat(uv_preds)
    rv_preds = torch.concat(rv_preds)

    mia_inputs = torch.concat([uv_preds, rv_preds])
    mia_labels = np.array([0] * uv_preds.shape[0] + [1]
                          * rv_preds.shape[0]).ravel()

    clf.fit(mia_inputs, mia_labels)

    for i in range(num_forget):
        uf_sample = [uf_pred[0] for uf_pred in uf_preds]
        rf_sample = [rf_pred[0] for rf_pred in rf_preds]

        uf_sample = torch.concat(uf_sample).numpy()
        rf_sample = torch.concat(rf_sample).numpy()
        H += example_epsilon(clf, uf_sample, rf_sample)
    return H / num_forget


def model_utility(ur_acc, ut_acc, rr_acc, rt_acc):
    return (ur_acc * rr_acc) / (ut_acc * rt_acc)


def competition(model: str, unlearn_path: str, retrain_path: str, retain, val, forget, test, device):
    model = getattr(models, model)
    unlearn_ckpts = [os.path.join(unlearn_path, ckpt)
                     for ckpt in os.listdir(unlearn_path)]
    retrain_ckpts = [os.path.join(retrain_path, ckpt)
                     for ckpt in os.listdir(retrain_path)]

    retain_loader = get_loader(retain)
    val_loader = get_loader(val)
    forget_loader = get_loader(forget)
    test_loader = get_loader(test)

    ur_preds, uv_preds, uf_preds = [], [], []
    rr_preds, rv_preds, rf_preds = [], [], []

    ur_acc, ut_acc, rr_acc, rt_acc = 0, 0, 0, 0

    # unlearn model
    for ckpt in unlearn_ckpts:
        net = model()
        net.load_state_dict(torch.load(ckpt))

        # retain set
        preds, acc = get_results(net, retain_loader, device)
        ur_preds.append(preds)
        ur_acc += acc

        # val set
        preds, acc = get_results(net, val_loader, device)
        uv_preds.append(preds)

        # forget set
        preds, acc = get_results(net, forget_loader, device)
        uf_preds.append(preds)

        # test set
        preds, acc = get_results(net, test_loader, device)
        ut_acc += acc

    # retrain model
    for ckpt in retrain_ckpts:
        net = model()
        net.load_state_dict(torch.load(ckpt))

        # retain set
        preds, acc = get_results(net, retain_loader, device)
        rr_preds.append(preds)
        rr_acc += acc

        # val set
        preds, acc = get_results(net, val_loader, device)
        rv_preds.append(preds)

        # forget set
        preds, acc = get_results(net, forget_loader, device)
        rf_preds.append(preds)

        # test set
        preds, acc = get_results(net, test_loader, device)
        rt_acc += acc

    utility_score = model_utility(ur_acc, ut_acc, rr_acc, rt_acc)
    forget_score = forget_quality(uv_preds, rv_preds, uf_preds, rf_preds)

    score = forget_score * utility_score
    return score
