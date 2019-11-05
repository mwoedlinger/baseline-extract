import torchvision
import torch
import torch.nn as nn


class LineFinder(nn.Module):

    def __init__(self):
        super(LineFinder, self).__init__()

        resnet = torchvision.models.resnet18(pretrained=True)

        for m in resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        layer0 = nn.Sequential(
            resnet.conv1,
            resnet.relu,
            resnet.maxpool)
        layer1 = resnet.layer1
        layer2 = resnet.layer2
        layer3 = resnet.layer3
        layer4 = resnet.layer4

        # out_channels = 4:
        # 0: x
        # 1: y
        # 2: angle
        # 3: box_size
        # 4: confidence
        self.model = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            layer4,#TODO: check if layer 4 should be used. layer 4 reduces the resolution to 32x32 patches
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        patch_size = 32.0

        out = self.mode(x)
        h = out.shape[2]
        w = out.shape[3]

        # Moves the coordinates from the small boxes to global coordinates
        # Not sure if needed. TODO: check if it makes a difference
        offset_tensor = torch.zeros(4, h, w)
        for row in range(offset_tensor.shape[1]):
            for column in range(offset_tensor.shape[2]):
                offset_tensor[0, row, column] = patch_size / 2.0 + row * patch_size
                offset_tensor[1, row, column] = patch_size / 2.0 + column * patch_size
                out[:, 4, row, column] = self.sigmoid(out[:, 4, row, column])

        affine_out = out + offset_tensor
        affine_out = nn.Flatten()(affine_out)

        return affine_out


    # use https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html to
    # match for loss