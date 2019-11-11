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
#                 # Possible solution: add together the point line distance and the distances of the point and
#                 # the line end points
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

        #TODO: find better solution
        if pred.shape[1] < label.shape[1]:
            label = label[:, 0:pred.shape[1], :]

        loss = 0


        for b in range(batch_size):
            # If the page is empty punish the model if it finds anything at all:
            if label.shape[1] == 0:
                conf_scores = pred[b, :, 4]
                log_c_anti = torch.log(1 - conf_scores + 0.00001)
                loss += log_c_anti.sum()
            else:
                # I get P predictions and T true labels.
                inp = pred[b, :, 0:4]
                targ = label[b, :, :]

                conf_scores = pred[b, :, 4]

                # Compute the confidence for all P predictions.
                log_c = torch.log(conf_scores + 0.00001)
                log_c_anti = torch.log(1 - conf_scores + 0.00001)

                # Expand such that for all T true lables I have a row of all predicted confidence logs.
                # The result is a P x T matrix.
                log_c_exp = log_c[:, None].expand(-1, targ.shape[0])
                log_c_anti_exp = log_c_anti[:, None].expand(-1, targ.shape[0])

                # Expand such that I get P x T x 4 matrices.
                inp_exp = inp[:, None, :].expand(-1, targ.shape[0], -1)
                targ_exp = targ[None, :, :].expand(inp.shape[0], -1, -1)

                # Compute the difference between every pair of prediction and true label locations.
                diff = (inp_exp[:, :, 0:4] - targ_exp[:, :, 0:4])
                normed_diff = torch.norm(diff, 2, 2) ** 2

                # Loss = Sum_{n=0}^N Sum_{m=0}^M    X_nm [alpha*MSE(l_n, p_m) - Log(c_m)] - (1- X_nm) Log(1-c_m)
                # where:
                #   N:      prediction dimension
                #   M:      label dimension
                #   X_mn:   linear assignement matrix
                #   l_n:    label coordinates
                #   p_m:    prediction coordinates
                #   c_m:    confidence scores

                # Compute the cost matrix. This is a P x T matrix.
                C = self.alpha * normed_diff - log_c_exp + log_c_anti_exp
                C = C.cpu().detach().numpy()

                X = torch.zeros(C.shape)
                x_c = torch.ones(C.shape[0])

                # For every column index (true), compute the row index (pred) where the cost is minimal.
                inp_idx, targ_idx = linear_sum_assignment(C)

                X[(inp_idx, targ_idx)] = 1.0
                x_c[inp_idx] = 0.0

                X = X.to(inp.device)
                x_c = x_c.to(inp.device)

                location_loss = (self.alpha * normed_diff * X).sum()
                confidence_loss = -(log_c_exp * X).sum() - (log_c_anti * x_c).sum()

                loss = location_loss + confidence_loss

        return loss

