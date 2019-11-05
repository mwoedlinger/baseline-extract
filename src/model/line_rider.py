import torch
import torch.nn as nn
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
        self.input_size = input_size

        # in: [N, 3, 32, 32] -> out: [N, 3]
        self.model_eyes = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=3)
        )

        # in: [N, 3, 32, 32] -> out: [N, 8]
        self.model_brain = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=8)
        )

        self.brain_hidden = nn.Linear(in_features=16, out_features=8)
        self.brain_output = nn.Linear(in_features=16, out_features=2)
        self.brain_activation = nn.Softmax(dim=1)

    def rider_eyes(self, x):
        """
        The model has 3 outputs:
            1:  The angle of the next baseline segment
            2:  The probability that the end of the baseline is reached i.e. that the next baseline segment
                is the last.
            3:  The length of the next baseline segment. Only relevant if it is the last segment.
        :param x:
        :return:
        """
        out = self.model_eyes(x)

        sina = nn.Tanh()(out[:, 0])
        cosa = nn.Tanh()(out[:, 1])

        return torch.cat([sina, cosa], dim=0)

    def rider_brain(self, x, hidden):
        """
        A RNN that determines when the end of the baseline is reached.
        :param x:
        :param hidden:
        :return:
        """
        cnn_out = self.model_brain(x)

        combined = torch.cat((cnn_out, hidden), dim=1)
        hidden = self.brain_hidden(combined)
        output = self.brain_output(combined)
        output = self.brain_activation(output)

        return output, hidden

    def forward(self, img, box_size, x_0=None, y_0=None, angle_0=None, baseline=None, reset_idx: int = 4):
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

        # The box is twice as wide as tall.
        box_width = box_size*2
        box_height = box_size

        x = x_0
        y = y_0
        angle = angle_0
        hidden = torch.tensor([0]*8).float().unsqueeze(0).to(self.device)

        # If baseline is provided extract start point and start angle from baseline
        if baseline is not None:
            x, y, angle = compute_start_and_angle(baseline, 0)
            x = x.to(self.device)
            y = y.to(self.device)
            angle = angle.to(self.device)

            train = True
        else:
            train = False

        sina = torch.sin(angle)
        cosa = torch.cos(angle)

        x_list = x.clone().unsqueeze(0)
        y_list = y.clone().unsqueeze(0)
        bl_end_list = torch.tensor(0.0).unsqueeze(0).to(self.device)

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

        idx = 0  # idx number 0 is the start point

        for _ in range(int(img_w / box_size) + 5):
            idx += 1
            alpha = angle

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
            x_s = -1.0 + x_scaled + box_width / 2 * torch.cos(alpha)
            y_s = -1.0 + y_scaled + box_width / 2 * torch.sin(alpha)

            # Theta describes an affine transformation and has the form
            # ( A_11, A_12, x_s)
            # ( A_21. A_22. y_s)
            # where A is the product of a rotation matrix and a scaling matrix and x_s, y_s describe the translation.
            theta_rot = torch.tensor(
                [[torch.cos(alpha), -torch.sin(alpha), x_s], [torch.sin(alpha), torch.cos(alpha), y_s], [0, 0, 1]])
            theta_scale = torch.tensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])
            theta = torch.mm(theta_rot, theta_scale)[0:2].unsqueeze(0).float()

            agrid = torch.nn.functional.affine_grid(theta, size).to(self.device)
            img_patch = torch.nn.functional.grid_sample(img, agrid, mode='nearest', padding_mode='zeros')

            out = self.rider_eyes(img_patch)
            end_out, hidden = self.rider_brain(img_patch, hidden=hidden)

            bl_end = end_out[0, 0]
            bl_end_length = end_out[0, 1]

            norm = out[0]**2 + out[1]**2


            # if bl_end < 0.5 the network predicted the end of the baseline.
            # The value of bl_end_length is then the quotient of box_size and the length of the final baseline segment.
            # cos(a + b) = cos(a)*cos(b) - sin(a)*sin(b)
            # sin(a + b) = sin(a)*cos(b) + cos(a)*sin(b)
            if train:
                if (reset_idx < 6 and len(baseline) < (idx + 1)) or (reset_idx >= 6 and bl_end > 0.5):
                    x = x + box_size * (cosa * out[0]/norm - sina * out[1]/norm) * bl_end_length
                    y = y - box_size * (sina * out[0]/norm + cosa * out[1]/norm) * bl_end_length

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
                    bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + box_size * (cosa * out[0]/norm - sina * out[1]/norm)
                    y = y - box_size * (sina * out[0]/norm + cosa * out[1]/norm)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
                    bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)
            else:
                if bl_end > 0.5:
                    x = x + box_size * (cosa * out[0]/norm - sina * out[1]/norm) * bl_end_length
                    y = y - box_size * (sina * out[0]/norm + cosa * out[1]/norm) * bl_end_length

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
                    bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + box_size * (cosa * out[0]/norm - sina * out[1]/norm)
                    y = y - box_size * (sina * out[0]/norm + cosa * out[1]/norm)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
                    bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)

            sina = out[1]/norm
            cosa = out[0]/norm
            angle += torch.atan(sina/cosa)

            # Every "reset_idx" step reset point and angle to the true label.
            if idx % reset_idx == reset_idx-1 and train:
                if len(baseline) > idx+1:
                    x, y, angle = compute_start_and_angle(baseline, idx, data_augmentation=True)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    angle = angle.to(self.device)

                    sina = torch.sin(angle)
                    cosa = torch.cos(angle)
                else:
                    break


        return torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0), bl_end_list