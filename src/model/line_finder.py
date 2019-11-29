import torchvision
import torch
import torch.nn as nn


class LineFinder(nn.Module):
    """
    The model has 4 out channels:
    0: x
    1: y
    2: angle
    3: confidence
    """
    def __init__(self, device: str):
        super(LineFinder, self).__init__()
        self.device = device
        self.out_dim = 4

        ################################
        resnet = torchvision.models.resnet34(pretrained=True)

        # for m in resnet.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()

        layer0 = nn.Sequential(
            resnet.conv1,
            resnet.relu,
            resnet.maxpool)
        layer1 = resnet.layer1
        layer2 = resnet.layer2
        layer3 = resnet.layer3
        # layer4 = resnet.layer4


        self.model = nn.Sequential(
            layer0,
            layer1,
            layer2,
            layer3,
            # layer4,
            nn.Conv2d(in_channels=256, out_channels=self.out_dim, kernel_size=1)
        )

        ########################################

        # vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        #
        # self.model = nn.Sequential(
        #     vgg16.features,
        #     nn.Conv2d(in_channels=512, out_channels=self.out_dim, kernel_size=1)
        # )

        self.sigmoid = nn.Sigmoid()
        # self.tanh = nn.Tanh()
        self.flatten_space = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        patch_size = 16.0
        # patch_size = 32.0

        out = self.model(x)

        h = out.shape[2]
        w = out.shape[3]
        batch_size = out.shape[0]

        # Moves the coordinates from the small boxes to global coordinates
        offset_tensor = torch.zeros(batch_size, self.out_dim, h, w)
        for b in range(batch_size):
            for row in range(h):
                for column in range(w):
                    offset_tensor[b, 0, row, column] = patch_size / 2.0 + row * patch_size
                    offset_tensor[b, 1, row, column] = patch_size / 2.0 + column * patch_size

                    out[b, -1, row, column] = self.sigmoid(out[b, -1, row, column])
                    out[b, 0, row, column] = patch_size/2.0 * self.tanh(out[b, 0, row, column]) #TODO: comment
                    out[b, 1, row, column] = patch_size/2.0 * self.tanh(out[b, 1, row, column]) #TODO: comment

        offset_tensor = offset_tensor.to(self.device)

        affine_out = out + offset_tensor
        affine_out = self.flatten_space(affine_out)
        affine_out = affine_out.permute(0, 2, 1)

        return affine_out