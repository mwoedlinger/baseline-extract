import torch
import torch.nn as nn
import random
import math
from ..utils.normalize_baselines import compute_start_and_angle


class LineRider(nn.Module):
    """
    A neural network that computes the baseline given the starting point, a scale parameter and an
    orientation parameter. It does this incrementally by applying a CNN to a rescaled image patch of a box around
    the last point. The box size is dependent on the letter size and needs to be given to the network as a parameter.
    The box is then resized to input_size x input_size.
    TLDR: input_size = number of pixels in width and height of the image patch that gets fed to the CNN
          box_size = size of the window that gets extracted (receptive field)
    """

    def __init__(self, device: str, input_size: int = 32):
        super(LineRider, self).__init__()
        self.device = device
        self.data_augmentation = True
        self.input_size = input_size

        # in: [N, 3, 32, 32] -> out: [N, 8]
        self.model_line = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=8)
        )
        # self.model_line = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=8)
        # )
        # self.model_line = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=8)
        # )

        # in: [N, 3, 32, 32] -> out: [N, 8]
        self.model_end = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 3), stride=(2, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 3), stride=(2, 1)),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, 14)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2)),
            nn.ReLU(),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(in_features=5 * 64, out_features=8)
        )
        # self.model_end = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=4),
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2),
        #     nn.ReLU(),
        #     nn.AvgPool2d(kernel_size=8),
        #     nn.Flatten(),
        #     nn.Linear(in_features=256, out_features=8)
        # )

        self.lin_out = nn.Linear(in_features=8, out_features=3)
        self.lin_out_end = nn.Linear(in_features=8, out_features=2)



    def rider_line(self, x):
        """
        A RNN for baseline extraction.
        The model has 2 outputs:
            0:  The sin of the angle of the next baseline segment.
            1:  The Cos of the angle of the next baseline segment.
        :param x:       Input
        :return:        A tuple containing the output and the hidden state both as a torch tensor
        """
        cnn_out = self.model_line(x)
        out = self.lin_out(cnn_out)

        sina = nn.Tanh()(out[:, 0])
        cosa = nn.Tanh()(out[:, 1])
        angle = nn.Tanh()(out[:, 2])*math.pi/2

        return torch.cat([sina, cosa, angle], dim=0)

    def rider_end(self, x):
        """
        A RNN for baseline extraction.
        The model has 2 outputs:
            0:  A propability for if the end of the baseline is reached.
            1:  The length of the next baseline segment (Only relevant if last segment).
        :param x:       Input
        :return:        A tuple containing the output and the hidden state both as a torch tensor
        """
        cnn_out = self.model_end(x)
        out = self.lin_out_end(cnn_out)

        bl_end = nn.Sigmoid()(out[:, 0])
        bl_end_length = nn.Sigmoid()(out[:, 1])

        return torch.cat([bl_end, bl_end_length], dim=0)

    def forward(self, img, box_size, sp=None, ep=None, angle_0=None, baseline=None, reset_idx: int = 4):
        """
        If a baseline is provided the model assumes training mode which means every fourth baseline point it will
        reset to the label given in 'baseline'. This also means that the computation graph is detached from
        previous computations every fourth baseline point.
        :param img:         The image as a torch tensor
        :param box_size:    The window size in the original image
        :param baseline:    The baseline as torch tensor
        :param reset_idx:   The index where the coordinates are reset to the true labels given in 'baseline'
                            For example if reset_idx = 4 every fourth window is reset to the true coordinates.
        :return:    a torch tensor with shape [n, 2] where n is the number of baseline points. [0,2] is the start point
                    of the baseline. [k, 0] is the x and [k, 1] is the y coordinate of point k.
        """

        # GET BASELINE, START AND END POINTS
        # The box is twice as wide as tall.
        box_width = box_size * 4
        box_height = box_size
        step_size = box_size * 2

        if baseline is not None:
            mode = 'baseline'
        elif ep is not None:
            mode = 'sp_ep'
        else:
            mode = 'sp'

        if mode == 'baseline':
            x, y, angle = compute_start_and_angle(baseline, 0, data_augmentation=self.data_augmentation)
            x = x.to(self.device)
            y = y.to(self.device)
            angle = angle.to(self.device)

            sp = torch.tensor([x, y]).to(self.device)
            ep = torch.tensor([baseline[-1, 0], baseline[-1, 1]]).to(self.device)
        elif mode == 'sp_ep':
            x = sp[0]
            y = sp[1]
            angle = angle_0

            max_dist = torch.sqrt(torch.pow(sp[0]-ep[0], 2) + torch.pow(sp[0]-ep[0], 2))
        elif mode == 'sp':
            x = sp[0]
            y = sp[1]
            angle = angle_0
        else:
            raise NotImplementedError

        # PREPARE OUTPUT LISTS
        patches = []
        x_list = x.clone().unsqueeze(0)
        y_list = y.clone().unsqueeze(0)
        bl_end_list = torch.tensor(0.0).unsqueeze(0).to(self.device)
        bl_end_length_list = torch.tensor(0.0).unsqueeze(0).to(self.device)

        # INITIALIZE EVERYTHING
        sina = torch.sin(angle)
        cosa = torch.cos(angle)

        img_w = img.size(2)
        img_h = img.size(3)
        w_box_ratio = box_width / img_w
        h_box_ratio = box_height / img_h

        # The otput size of the grid i.e. the size of the tensor that is fed into the network:
        size = (1, 3, self.input_size, self.input_size)

        # grid_sample expects grid coordinates scaled to [-1,1]. This means that (-1,-1) is the top left corner and
        # (1,1) is the bottom right corner.
        w_box = box_width / img_w * 2
        h_box = box_height / img_h * 2

        # Distinguishing between scale_x and scale_y is actually not necessary for squared images
        # I just left it in there in case i drop the resize to square
        scale_x = w_box_ratio
        scale_y = h_box_ratio

        # LOOP OVER THE BASELINE
        for idx in range(1, int(img_w / box_size) + 5):
            alpha = angle

            # grid_sample expects grid coordinates scaled to [-1,1]. This means that (-1,-1) is the top left corner and
            # (1,1) is the bottom right corner.
            x_scaled = x / img_w * 2
            y_scaled = y / img_h * 2

            # The window is taken from the middle of the image:
            # o Move the top left corner with - (1,1)
            # o Move to specified point with + (x_scaled, y_scaled)
            # o move such that the start point is in the middle of the right border with
            #   + (cos(angle) * w_box, -sin(angle) * w_box) + (0, -h_box/4
            # // This assumes that the image is squared, otherwise the hypotenuse is not exactly w_box/2
            x_s = -1.0 + x_scaled + w_box / 2 * torch.cos(alpha)
            y_s = -1.0 + y_scaled - w_box / 2 * torch.sin(alpha)# - h_box/4 #TODO: leave or comment out?

            # Theta describes an affine transformation and has the form
            # ( A_11, A_12, x_s)
            # ( A_21. A_22. y_s)
            # where A is the product of a rotation matrix and a scaling matrix and x_s, y_s describe the translation.
            # The angle is set to -alpha because the rotation of the original image must be reversed.
            theta_rot = torch.tensor(
                [[torch.cos(-alpha), -torch.sin(-alpha), x_s], [torch.sin(-alpha), torch.cos(-alpha), y_s], [0, 0, 1]])
            theta_scale = torch.tensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
            theta = torch.mm(theta_rot, theta_scale)[0:2].unsqueeze(0).float()

            agrid = torch.nn.functional.affine_grid(theta, size).to(self.device)
            img_patch = torch.nn.functional.grid_sample(img, agrid, mode='nearest', padding_mode='zeros')
            patches.append(img_patch)

            # out, hidden = self.rider_line(img_patch, hidden=hidden)
            out = self.rider_line(img_patch)
            if mode in ['baseline', 'sp']:
                out_end = self.rider_end(img_patch)

                bl_end = out_end[0]
                bl_end_length = out_end[1]
                bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)
                bl_end_length_list = torch.cat([bl_end_length_list, bl_end_length.unsqueeze(0)], dim=0)
            else:
                out_end = torch.tensor([0.0, 0.0]).to(self.device)

                bl_end = out_end[0]
                bl_end_length = out_end[1]

            norm = torch.sqrt(out[0]**2 + out[1]**2)
            # sina_new = out[0]/norm
            # cosa_new = out[1]/norm
            angle_out = out[2]
            sina_new = (out[0]/norm + torch.sin(angle_out))/2
            cosa_new = (out[1]/norm + torch.cos(angle_out))/2

            ################################### TODO: test if this works better:
            # x_out = out[0]
            # y_out = out[1]
            # angle_out = out[2]
            #
            # dx = step_size * (torch.sin(angle)*y_out + torch.cos(angle)*x_out)
            # dy = step_size * (torch.sin(angle)*y_out + torch.cos(angle)*y_out)
            #
            # angle = angle + (angle_out + torch.atan(y_out/x_out)) TODO: necessary?
            ###################################

            # cos(a + b) = cos(a)*cos(b) - sin(a)*sin(b)
            # sin(a + b) = sin(a)*cos(b) + cos(a)*sin(b)
            if mode == 'sp':
                # if bl_end > 0.8 the network predicted the end of the baseline.
                # The value of bl_end_length is then the quotient of box_size and the length of the final baseline segment.
                if bl_end > 0.8:
                    x = x + step_size * (cosa * cosa_new - sina * sina_new) * bl_end_length
                    y = y - step_size * (sina * cosa_new + cosa * sina_new) * bl_end_length

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + step_size * (cosa * cosa_new - sina * sina_new)
                    y = y - step_size * (sina * cosa_new + cosa * sina_new)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

            elif mode == 'sp_ep':
                x = x + step_size * (cosa * cosa_new - sina * sina_new)
                y = y - step_size * (sina * cosa_new + cosa * sina_new)

                if torch.sqrt(torch.pow(sp[0] - x, 2) + torch.pow(sp[1] - y, 2)) > max_dist:
                    x_list = torch.cat([x_list, ep[0].unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, ep[1].unsqueeze(0)], dim=0)

                    break
                else:
                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

            elif mode == 'baseline':
                # if (reset_idx < 6 and len(baseline) <= (idx + 1)) or (reset_idx >= 6 and bl_end > 0.5):
                if len(baseline) == (idx + 1):
                    x = baseline[-1, 0]
                    y = baseline[-1, 1]

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + step_size * (cosa * cosa_new - sina * sina_new)
                    y = y - step_size * (sina * cosa_new + cosa * sina_new)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
            else:
                raise NotImplementedError

            sina = sina_new
            cosa = cosa_new
            # angle += torch.atan(sina / cosa)
            angle += (torch.atan(sina / cosa) + angle_out)/2

            if self.data_augmentation == True:
                x_range = max(2, int(torch.sin(angle) * box_size/6))
                y_range = max(2, int(torch.cos(angle) * box_size/6))
                x += random.randint(-x_range, x_range)
                y += random.randint(-y_range, y_range)
                angle += random.uniform(-0.3, 0.3)

            if mode == 'baseline':
                # Every "reset_idx" step reset point and angle to the true label.
                if idx % reset_idx == reset_idx-1:
                    if len(baseline) > idx+1:
                        x, y, angle = compute_start_and_angle(baseline, idx, data_augmentation=self.data_augmentation)
                        x = x.to(self.device)
                        y = y.to(self.device)
                        angle = angle.to(self.device)

                        sina = torch.sin(angle)
                        cosa = torch.cos(angle)
                    else:
                        print('ERROR! This message should not be reached!')
                        break

        return torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0), bl_end_list, \
               bl_end_length_list, patches