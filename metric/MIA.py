import numpy as np
from sklearn import linear_model, model_selection
import torch.nn as nn


def compute_losses(net, loader, device: str = "cpu"):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    for batch_data in loader:
        inputs, targets = batch_data['image'], batch_data['age']
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

    return np.array(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Computes cross-validation score of a membership inference attack.

    Args:
      sample_loss : array_like of shape (n,).
        objective function evaluated on n samples.
      members : array_like of shape (n,),
        whether a sample was used for training.
      n_splits: int
        number of splits to use in the cross-validation.
    Returns:
      scores : array_like of size (n_splits,)
    """

    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def mia(model, test_loader, forget_loader):
    rt_test_losses = compute_losses(model, test_loader)
    rt_forget_losses = compute_losses(model, forget_loader)

    rt_samples_mia = np.concatenate(
        (rt_test_losses, rt_forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(rt_test_losses) + [1] * len(rt_forget_losses)
    mia_score = simple_mia(rt_samples_mia, labels_mia)
    return mia_score.mean()
