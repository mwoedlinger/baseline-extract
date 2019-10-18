from collections import OrderedDict
import torch
import torch.nn as nn

class LineRider(nn.Module):
    """
    A neural network that computes the baseline given the starting point, a scale parameter and an
    orientation parameter. It does this incrementally by applying a CNN to a rescaled image patch of a box around
    the last point. The box size is dependent on the letter size and needs to be given to the network as a parameter.
    The box is then resized to 32x32.
    """

    def __init__(self):
        super(LineRider, self).__init__()
        self.input_size = 32

        self.rider = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5, stride=2)),
            ('conv2', nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1)),
            ('maxPool', nn.MaxPool2d(kernel_size=3, stride=2)),
            ('conv3', nn.Conv2d(in_channels=10, out_channels=15, kernel_size=4, stride=1)),
            ('conv4', nn.Conv2d(in_channels=15, out_channels=2, kernel_size=1, stride=1))
            #TODO: ^^ to 3 output channels for: baseline end, baseline length (only relevant if baseline end) and angle
        ]))



    def forward(self, img, x_0, y_0, angle, box_size):
        angle = angle
        x = x_0
        y = y_0

        img_w = img.size(1)
        img_h = img.size(2)

        w_box_ratio = box_size/img_w
        h_box_ratio = box_size/img_h

        #The size of the windows:
        size = (1, 3, self.input_size, self.input_size)

        # grid_sample expects grid coordinates scaled to [-1,1]. This means that (-1,-1) is the top left corner and
        # (1,1) is the bottom right corner.
        w_box = box_size/img_w*2
        h_box = box_size/img_h*2

        # Distinguishing between scale_x and scale_y is actually not necessary for squared images
        # I just left it in there in case i drop the resize to square
        scale_x = w_box_ratio
        scale_y = h_box_ratio


        baseline_end = 0

        while(baseline_end < 0.5):
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

            agrid = torch.nn.functional.affine_grid(theta, size, align_corners=None)
            img_patch = torch.nn.functional.grid_sample(img, agrid, mode='nearest', padding_mode='zeros',
                                                        align_corners=None)

            out = self.rider(img_patch)

            baseline_end = out[0]
            angle = out[1]

            x = box_size*torch.sin(angle)
            y = box_size*torch.cos(angle)







