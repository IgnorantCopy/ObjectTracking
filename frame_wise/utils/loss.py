import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, pred, label):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(pred, label)
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss