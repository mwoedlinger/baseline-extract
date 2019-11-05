import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from ..utils.distances import point_line_distance


class L12Loss(nn.Module):
    def __init__(self):
        super(L12Loss, self).__init__()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, target):
        return self.l1_loss(input, target) + self.l2_loss(input, target)

# class BaselineLoss(nn.Module):
#     def __init__(self):
#         super(BaselineLoss, self).__init__()
#
#     def forward(self, input, target):
#         N = input.shape[1]
#         M = target.shape[1]-1
#
#
#         cost = torch.zeros(N, M)
#
#         for n in range(N):
#             for m in range(M):
#                 x_0, y_0 = input[n]
#                 xp_1, yp_1 = target[m]
#                 xp_2, yp_2 = target[m+1]
#
#                 # WRONG: maybe a point lies directly on a line (continued beyond the end points) but
#                 # the the point and the line are not matched in the way intended.
#                 cost[n, m] = point_line_distance(x_0, y_0, xp_1, yp_1, xp_2, yp_2)
#
#         inp_idx, target_idx = linear_sum_assignment(cost)


class LineFinderLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super(LineFinderLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, input, target):
        N = input.shape[1]
        M = target.shape[1]

        cost = torch.zeros(N, M)

        for n in range(N):
            for m in range(M):
                cost[n, m] = self.mse(input[n, 0:2], target[m, 0:2])

        inp_idx, target_idx = linear_sum_assignment(cost)

        X = torch.zeros(N, M)
        for p_idx in inp_idx:
            for l_idx in target_idx:
                X[p_idx, l_idx] = 1

        # Loss = Sum_{n=0}^N Sum_{m=0}^M    X_nm [alpha*MSE(l_n, p_m) - Log(c_m)] - (1- X_nm) Log(1-c_m)
        # where:
        #   N:      prediction dimension
        #   M:      label dimension
        #   X_mn:   linear assignement matrix
        #   l_n:    label coordinates
        #   p_m:    prediction coordinates
        #   c_m:    confidence scores

        loss = 0
        for n in range(N):
            for m in range(M):
                loss += X[n, m]*(self.alpha*self.mse(input, target) - torch.log(input[n, 4]))\
                        - (1-X[n, m])*(torch.log(1-input[n, 4]))
        loss += nn.MSELoss()(input[:, 3], target)

