import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=0.1) -> None:
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.kl_criterion = nn.KLDivLoss(reduction="batchmean")
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, baseline, targets):
        kl_loss = self.kl_criterion(inputs, baseline)
        ce_loss = self.ce_criterion(inputs, targets)

        loss = ce_loss + self.alpha * kl_loss
        return loss
