import torch
import torch.nn as nn
from ..utils.distances import *


class L12Loss(nn.Module):
    def __init__(self):
        super(L12Loss, self).__init__()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, input, target):
        return self.l1_loss(input, target) + self.l2_loss(input, target)

def prepare_data_for_loss(gt_baseline, pred_baseline):
    """
    Prepare the labels for the loss function. There are three components:
       o) A regression loss for the predicted baseline points
       o) A binary crossentropy loss for the baseline end prediction
    """
    # Match the length. In case the network didn't find the ending properly we add the last predicted
    # point repeatedly to match the dimension of bl_n

    # Baseline label
    baseline_label = [gt_baseline[0]]

    for pred_point in pred_baseline[1:-1]:
        gt_point, _ = get_closest_point_on_polygon(pred_point, gt_baseline)
        try: # DEBUG!!!
            gt_point = torch.stack(gt_point)
        except Exception as e:
            print('EXCPETION THROWN!')
            print('pred_point = {}; gt_baseline = {}'.format(pred_point, gt_baseline))
            print('closest_point = {}'.format(get_closest_point_on_polygon(pred_point, gt_baseline)[0]))
            print(e)
            raise e
        baseline_label.append(gt_point)

    baseline_label.append(gt_baseline[-1])
    baseline_label = torch.stack(baseline_label)

    # Baseline end label
    sigma = 30
    x_end, y_end = gt_baseline[-1][0], gt_baseline[-1][1]
    end_label = torch.tensor([normal(pred_baseline[k][0], pred_baseline[k][1], x_end, y_end, sigma)
                              for k in range(0, len(pred_baseline))])

    return baseline_label, end_label

# def prepare_data_for_loss(bl_n, c_list, bl_end_list, bl_n_end_length, bl_end_length_list, box_size, device):
#     """
#     Prepare the labels for the loss function. There are three components:
#        o) A regression loss for the predicted baseline points
#        o) A binary crossentropy loss for the baseline end prediction
#        o) A regression loss for predicting the length of the last baseline segment.
#     """
#     # Match the length. In case the network didn't find the ending properly we add the last predicted
#     # point repeatedly to match the dimension of bl_n
#     if len(c_list) < len(bl_n):
#         diff = len(bl_n) - len(c_list)
#         c_list = torch.cat([c_list, torch.stack([c_list[-1]] * diff)], dim=0)
#     elif len(c_list) > len(bl_n):
#         diff = len(c_list) - len(bl_n)
#         bl_n = torch.cat([bl_n, torch.stack([bl_n[-1]] * diff)], dim=0)
#
#     l_label = bl_n[:-1]
#     l_pred = c_list[:-1]
#
#     l_bl_end_label = torch.tensor([normal(bl_n[k][0], bl_n[k][1], bl_n[-1][0], bl_n[-1][1], box_size)
#                                    for k in range(0, len(bl_n))]).to(device)
#     l_bl_end_pred = bl_end_list
#
#     if len(l_bl_end_pred) < len(l_bl_end_label):
#         diff = len(l_bl_end_label) - len(l_bl_end_pred)
#         l_bl_end_pred = torch.cat([l_bl_end_pred, torch.stack([torch.tensor(0.0).to(device)] * diff)], dim=0)
#     elif len(l_bl_end_pred) > len(l_bl_end_label):
#         diff = len(l_bl_end_pred) - len(l_bl_end_label)
#         l_bl_end_label = torch.cat([l_bl_end_label, torch.stack([torch.tensor(1.0).to(device)] * diff)], dim=0)
#
#     l_bl_end_length_label = bl_n_end_length
#     l_bl_end_length_pred = bl_end_length_list[-1]
#
#     return l_label, l_pred, l_bl_end_label, l_bl_end_pred, l_bl_end_length_label, l_bl_end_length_pred
