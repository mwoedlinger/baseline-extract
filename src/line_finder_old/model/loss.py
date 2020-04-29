import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class LineFinderLoss(nn.Module):
    """
    Loss = Sum_{n=0}^N Sum_{m=0}^M    X_nm [alpha*MSE(l_n, p_m) - Log(c_m)] - (1- X_nm) Log(1-c_m)
    where:
      N:      prediction dimension
      M:      label dimension
      X_mn:   linear assignement matrix
      l_n:    label coordinates
      p_m:    prediction coordinates
      c_m:    confidence scores
    """
    def __init__(self, alpha=0.1):
        super(LineFinderLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, label, label_len):
        batch_size = pred.shape[0]

        #find better solution
        if pred.shape[1] < label.shape[1]:
            label = label[:, 0:pred.shape[1], :]

        location_loss = 0
        confidence_loss = 0

        for b in range(batch_size):
            # If the page is empty punish the model if it finds anything at all:
            if label.shape[1] == 0:
                conf_scores = pred[b, :, -1]
                confidence_loss += -torch.log(1 - conf_scores + 0.01).sum()
            else:
                # I get P predictions and T true labels.
                inp = pred[b, :, 0:3]
                targ = label[b, 0:label_len[b], :]

                conf_scores = pred[b, :, -1]

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
                diff = (inp_exp[:, :, 0:3] - targ_exp[:, :, 0:3])
                normed_diff = torch.norm(diff, 2, 2) ** 2

                # Compute the cost matrix. This is a P x T matrix.
                C = self.alpha * normed_diff/2.0 - log_c_exp + log_c_anti_exp
                C = C.cpu().detach().numpy()

                X = torch.zeros(C.shape)
                x_c = torch.ones(C.shape[0])

                # For every row index (true), compute the column index (pred) where the cost is minimal.
                row_idx, col_idx = linear_sum_assignment(C.T)

                X[(col_idx, row_idx)] = 1.0
                x_c[col_idx] = 0.0

                X = X.to(inp.device)
                x_c = x_c.to(inp.device)

                location_loss += (normed_diff * X).sum()/2.0
                confidence_loss += -(log_c_exp * X).sum() - (log_c_anti * x_c).sum()+0.0001

        loss = self.alpha*location_loss + confidence_loss

        return loss, self.alpha * location_loss, -(log_c_exp * X).sum(), -(log_c_anti * x_c).sum()