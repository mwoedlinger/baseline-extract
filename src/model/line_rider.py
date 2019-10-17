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



    def forward(self, img, x_0, y_0, angle, scale):

