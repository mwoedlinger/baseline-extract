import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import time
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

    def forward(self, pred, label):
        batch_size = pred.shape[0]
        n_tot = pred.shape[1]
        m_tot = label.shape[1]

        loss = 0

        for b in range(batch_size):

            inp = pred[b, :, 0:4]
            targ = label[b, :, :]

            conf_scores = pred[b, :, 4]

            log_c = torch.log(conf_scores + 0.00001)
            log_c_anti = torch(1 - conf_scores + 0.00001)

            log_c_exp = log_c[:, None].expand(-1, targ.shape[0])
            log_c_anti_exp = log_c_anti[:, None].expand(-1, targ.shape[0])
            inp_exp = inp[:, None, :].expand(-1, targ.shape[0], -1)
            targ_exp = targ[None, :, :].expand(inp.shape[0], -1, -1)

            diff = (inp_exp - targ_exp)
            normed_diff = torch.norm(diff, 2, 3) ** 2

            # Loss = Sum_{n=0}^N Sum_{m=0}^M    X_nm [alpha*MSE(l_n, p_m) - Log(c_m)] - (1- X_nm) Log(1-c_m)
            # where:
            #   N:      prediction dimension
            #   M:      label dimension
            #   X_mn:   linear assignement matrix
            #   l_n:    label coordinates
            #   p_m:    prediction coordinates
            #   c_m:    confidence scores

            C = self.alpha*normed_diff - log_c_exp + log_c_anti_exp

            X = torch.zeros(C.shape)

            inp_idx, targ_idx = linear_sum_assignment(C)














            # cost = torch.zeros(n_tot, m_tot)
            #
            # for n in range(n_tot):
            #     for m in range(m_tot):
            #         cost[n, m] = self.mse(pred[b, n, 0:2], label[b, m, 0:2])
            #
            # cost = cost.to('cuda:3')
            # inp_idx, target_idx = linear_sum_assignment(cost.detach())
            #
            # X = torch.zeros(n_tot, m_tot)
            #
            # for p_idx in inp_idx:
            #     for l_idx in target_idx:
            #         X[p_idx, l_idx] = 1
            #
            #
            # for m in range(m_tot):
            #     loss += self.alpha * cost[inp_idx[m], target_idx[m]] - torch.log(pred[b, inp_idx[m], 4]) \
            #             - torch.log(1 - pred[b, inp_idx[m], 4])#substract and add in next step
            #
            # for n in range(n_tot):
            #     loss += torch.log(1 - pred[b, n, 4])
            #
            # loss += self.alpha * nn.MSELoss()(pred[b, inp_idx, 3], label[b, target_idx, 3])

        return loss

