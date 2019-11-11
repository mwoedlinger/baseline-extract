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
        self.model = nn.Sequential(
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
            nn.Linear(in_features=128, out_features=8)
        )

        self.lin_out = nn.Linear(in_features=16, out_features=4)
        self.lin_hidden = nn.Linear(in_features=16, out_features=8)

    def rider_eyes(self, x, hidden):
        """
        A RNN for baseline extraction.
        The model has 3 outputs:
            0:  The sin of the angle of the next baseline segment.
            1:  The Cos of the angle of the next baseline segment.
            2:  A propability for if the end of the baseline is reached.
            3:  The length of the next baseline segment (Only relevant if last segment).
        :param x:       Input
        :param hidden:  The hidden State
        :return:        A tuple containing the output and the hidden state both as a torch tensor
        """
        cnn_out = self.model(x)

        combined = torch.cat((cnn_out, hidden), dim=1)
        hidden = self.lin_hidden(combined)
        out = self.lin_out(combined)

        sina = nn.Tanh()(out[:, 0])
        cosa = nn.Tanh()(out[:, 1])
        bl_end = nn.Sigmoid()(out[:, 2])
        bl_end_length = nn.Sigmoid()(out[:, 3])

        return torch.cat([sina, cosa, bl_end, bl_end_length], dim=0), hidden

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
            x, y, angle = compute_start_and_angle(baseline, 0, data_augmentation=True)
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
        bl_end_length_list = torch.tensor(0.0).unsqueeze(0).to(self.device)

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
            y_s = -1.0 + y_scaled - w_box / 2 * torch.sin(alpha) - h_box/4 #TODO: leave or comment out?

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

            out, hidden = self.rider_eyes(img_patch, hidden=hidden)

            norm = torch.sqrt(out[0]**2 + out[1]**2)
            sina_new = out[0]/norm
            cosa_new = out[1]/norm

            bl_end = out[2]
            bl_end_length = out[3]
            bl_end_list = torch.cat([bl_end_list, bl_end.unsqueeze(0)], dim=0)
            bl_end_length_list = torch.cat([bl_end_length_list, bl_end_length.unsqueeze(0)], dim=0)

            # if bl_end < 0.5 the network predicted the end of the baseline.
            # The value of bl_end_length is then the quotient of box_size and the length of the final baseline segment.
            # cos(a + b) = cos(a)*cos(b) - sin(a)*sin(b)
            # sin(a + b) = sin(a)*cos(b) + cos(a)*sin(b)
            if train:
                # if (reset_idx < 6 and len(baseline) <= (idx + 1)) or (reset_idx >= 6 and bl_end > 0.5):
                if len(baseline) == (idx + 1):
                    x = baseline[-1, 0]
                    y = baseline[-1, 1]

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + box_size * (cosa * cosa_new - sina * sina_new)
                    y = y - box_size * (sina * cosa_new + cosa * sina_new)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)
            else:
                if bl_end > 0.5:
                    x = x + box_size * (cosa * cosa_new - sina * sina_new)
                    y = y - box_size * (sina * cosa_new + cosa * sina_new)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

                    break
                else:
                    x = x + box_size * (cosa * cosa_new - sina * sina_new)
                    y = y - box_size * (sina * cosa_new + cosa * sina_new)

                    x_list = torch.cat([x_list, x.unsqueeze(0)], dim=0)
                    y_list = torch.cat([y_list, y.unsqueeze(0)], dim=0)

            sina = sina_new
            cosa = cosa_new
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
                    print('ERROR! This message should not be reached!')
                    break

        return torch.cat([x_list.unsqueeze(0), y_list.unsqueeze(0)], dim=0).permute(1, 0), bl_end_list, bl_end_length_list