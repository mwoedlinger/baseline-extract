from collections import OrderedDict
import math
import torchvision
import torch
import torch.nn as nn


class LineRider(nn.Module):
    """
    A neural network that computes the baseline given the starting point, a scale parameter and an
    orientation parameter. It does this incrementally by applying a CNN to a rescaled image patch of a box around
    the last point. The box size is dependent on the letter size and needs to be given to the network as a parameter.
    The box is then resized to input_size x input_size.
    TLDR: input_size = number of pixels in width and height of the image patch that gets fed to the CNN
          box_size = size of the window that gets extracted (receptive field)
    """

    def __init__(self, device: str, input_size: int = 32, reset_idx: int = 4):
        super(LineRider, self).__init__()
        self.device = device
        self.input_size = input_size
        self.reset_idx = reset_idx

        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1)),
            ('relu2', nn.ReLU()),
            ('maxPool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(in_channels=10, out_channels=15, kernel_size=4, stride=1)),
            ('relu3', nn.ReLU()),
            # ('conv4', nn.Conv2d(in_channels=15, out_channels=2, kernel_size=1, stride=1)),
            ('flatten', nn.Flatten()),
            ('fc', nn.Linear(in_features=15, out_features=2))
            #TODO: ^ to 3 output channels for: baseline end, baseline length (only relevant if baseline end) and angle
        ]))

        # # mnasnet is a light image classifier trained on imageNet
        # self.model = torchvision.models.mnasnet1_0(pretrained=True)
        #
        # for m in self.model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #
        # self.model.classifier = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(in_features=1280, out_features=3))

    def rider(self, x):
        out = self.model(x)

        # cosa = nn.Tanh()(out[:, 0])
        # sina = nn.Tanh()(out[:, 1])
        # bl_end = out[:, 2]
        #
        # return torch.cat([cosa, sina, bl_end], dim=0)

        angle = nn.Tanh()(out[:, 0])*math.pi/2
        bl_end = nn.ReLU()(out[:, 1])

        return torch.cat([angle, bl_end], dim=0)

    def compute_start_and_angle(self, baseline, idx):
        if torch.abs(baseline[idx, 0] - baseline[idx + 1, 0]) < 0.001:
            angle = torch.tensor(math.pi / 2.0).to(self.device)
        else:
            angle = torch.atan((baseline[idx, 1] - baseline[idx + 1, 1]) / (baseline[idx, 0] - baseline[idx + 1, 0]))
        # TODO: ^ make sure that the angle doesn't flip for vertical baselines

        x = baseline[idx, 0]
        y = baseline[idx, 1]

        return x, y, angle

    def forward(self, img, box_size, baseline=None):
        """
        If a baseline is provided the model assumes training mode which means every fourth baseline point it will
        reset to the label given in 'baseline'. This also means that the computation graph is detached from
        previous computations every fourth baseline point.
        :param img:
        :param box_size:
        :param baseline:
        :return:    a torch tensor with shape [n, 2] where n is the number of baseline points. [0,2] is the start point
                    of the baseline. [k, 0] is the x and [k, 1] is the y coordinate of point k.
        """

        x, y, angle = self.compute_start_and_angle(baseline, 0)

        x_list = x.clone().unsqueeze(0)
        y_list = y.clone().unsqueeze(0)

        img_w = img.size(2)
        img_h = img.size(3)

        w_box_ratio = box_size / img_w
        h_box_ratio = box_size / img_h

        # The size of the windows:
        size = (1, 3, self.input_size, self.input_size)

        # grid_sample expects grid coordinates scaled to [-1,1]. This means that (-1,-1) is the top left corner and
        # (1,1) is the bottom right corner.
        w_box = box_size / img_w * 2
        h_box = box_size / img_h * 2

        # Distinguishing between scale_x and scale_y is actually not necessary for squared images
        # I just left it in there in case i drop the resize to square
        scale_x = w_box_ratio
        scale_y = h_box_ratio

        baseline_end = 0
        idx = 1  # idx number 0 is the start point

        for _ in range(int(img_w / box_size) + 5):
            alpha = -angle

            # grid_sample expects grid coordinates scaled to [-1,1]. This means that (-1,-1) is the top left corner and
            # (1,1) is the bottom right corner.
            x_scaled = x / img_w * 2
            y_scaled = y / img_h * 2

            # The window is taken from the middle of the image:
            # o Move the top left corner with - (1,1)
            # o Move to specified point with + (x_scaled, y_scaled)
            # o move such that the start point is in the middle of the right border with
            #   + (cos(angle) * w_box, -sin(angle) * w_box)
            # // This assumes that the image is squared, otherwise the hypotenuse is not exactly w_box/2
            x_s = -1.0 + x_scaled + torch.cos(angle) * w_box / 2
            y_s = -1.0 + y_scaled - torch.sin(angle) * w_box / 2

            # Theta describes an affine transformation and has the form
            # ( A_11, A_12, x_s)
            # ( A_21. A_22. y_s)
            # where A is the product of a rotation matrix and a scaling matrix and x_s, y_s describe the translation.
            theta = torch.tensor([[scale_x * torch.cos(alpha), -scale_x * torch.sin(alpha), x_s],
                                  [scale_y * torch.sin(alpha), scale_y * torch.cos(alpha), y_s]]).unsqueeze(0).float()

            agrid = torch.nn.functional.affine_grid(theta, size).to(self.device)
            img_patch = torch.nn.functional.grid_sample(img, agrid, mode='nearest', padding_mode='zeros')

            out = self.rider(img_patch)

            angle = angle + out[0]
            bl_end = out[1]

            # if out[2] < 1 the network predicted the end of the baseline.
            # The value of out[2] is then the quotient of the
            if False:#bl_end < 1:
                x = x + box_size * torch.cos(angle) * bl_end
                y = y + box_size * torch.sin(angle) * bl_end

                x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

                break
            else:
                x = x + box_size * torch.cos(angle)
                y = y + box_size * torch.sin(angle)

                x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

            # # cos(a + b) = cos(a)*cos(b) - sin(a)*sin(b)
            # x = x + box_size * (cosa*out[0] - sina*out[1])
            # # sin(a + b) = sin(a)*cos(b) + cos(a)*sin(b)
            # y = y + box_size * (sina*out[0] + cosa*out[1])
            #
            # sina = out[1]
            # cosa = out[0]
            #
            # angle = angle + torch.atan(cosa/sina)
            # # If this ^ version is used add these two lines to the beginning of the program:
            # # sina = torch.sin(angle)
            # # cosa = torch.cos(angle)


            if idx % self.reset_idx == self.reset_idx-1:
                if len(baseline) > (idx+1):
                    x, y, angle = self.compute_start_and_angle(baseline, idx)
                else:
                    break

            idx += 1
            # TODO: delete
            if len(baseline) < (idx + 1):
                break


        return torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0)