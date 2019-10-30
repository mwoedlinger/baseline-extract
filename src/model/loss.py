import torch
import torch.nn as nn


class L12Loss(nn.Module):
    def __init__(self):
        super(L12Loss, self).__init__()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, target):
        return self.l1_loss(input, target) + self.l2_loss(input, target)

class LineFinderLoss(nn.Module):
    def __init__(self):
        super(LineFinderLoss, self).__init__()

    def forward(self, input, target):
        pass