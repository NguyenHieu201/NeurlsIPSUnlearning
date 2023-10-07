import numpy as np
from sklearn import linear_model, model_selection
import torch.nn as nn
import typer
import torch


@torch.no_grad()
def compute_losses(net, loader, device: str = "cpu"):
    """Auxiliary function to compute per-sample losses"""

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []

    net.eval()
    net = net.to(device)
    cnt = 0
    total_sample = 0
    for batch_data in loader:
        inputs, targets = batch_data['image'], batch_data['age']
        inputs, targets = inputs.to(device), targets.to(device)

        logits = net(inputs)
        losses = criterion(logits, targets).numpy(force=True)
        total_sample += inputs.shape[0]
        cnt += (logits.argmax(axis=1) == targets).sum().item()
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


def mia(model, test_loader, forget_loader, device: str = "cpu"):
    rt_test_losses = compute_losses(model, test_loader, device)
    rt_forget_losses = compute_losses(model, forget_loader, device)

    num_choose_values = 3000
    mia_score = 0
    repeat_times = 10
    for i in range(repeat_times):
        np.random.seed(i)
        rt_test_choice = np.random.choice(
            rt_test_losses, num_choose_values, False)
        rt_forget_choice = np.random.choice(
            rt_forget_losses, num_choose_values, False)
        rt_samples_mia = np.concatenate(
            (rt_test_choice, rt_forget_choice)).reshape((-1, 1))
        labels_mia = [0] * num_choose_values + [1] * num_choose_values
        mia_score += simple_mia(rt_samples_mia, labels_mia).mean()
    return mia_score / repeat_times
